/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cassert>
#include <string>
#include <stdexcept>

#include "model/topology.hpp"
#include "model/network-legacy.hpp"
#include "model/network-factory.hpp"

bool gSparseEvaluate = true;

namespace model
{

//--------------------------------------------//
//              Topology::Specs               //
//--------------------------------------------//

void Topology::Specs::ParseAccelergyERT(config::CompoundConfigNode ert)
{
  // std::cout << "Replacing energy numbers..." << std::endl;
  assert(ert.exists("tables"));
  assert(ert.exists("version"));  
  double ertVersion;
  // check the version of the ERT
  ert.lookupValue("version", ertVersion);
  config::CompoundConfigNode formattedErt;
  if (ertVersion == 0.2) {
    // no additional formatting needed
    formattedErt = ert;
  } else if (ertVersion == 0.3) {
    auto table = ert.lookup("tables");
    assert(table.isList());
    YAML::Node root;
    std::string componentName;
    std::string actionName;
    config::CompoundConfigNode componentActionList;
    for(int i = 0; i < table.getLength(); i++){
      assert(table[i].exists("name"));
      assert(table[i].exists("actions"));
      table[i].lookupValue("name", componentName);
      componentActionList = table[i].lookup("actions");
      for (int i = 0; i < componentActionList.getLength(); i++){
        assert(componentActionList[i].exists("name"));
        componentActionList[i].lookupValue("name", actionName);
        root["tables"][componentName][actionName].push_back(componentActionList[i].getYNode());
      }
    }
    formattedErt = config::CompoundConfigNode(nullptr, root);
  } else {
    std::cout << "Invalid Accelergy ERT version: " << ertVersion << std::endl;
    assert(false);
  }
  // parsing 
  std::vector<std::string> keys;
  auto table = formattedErt.lookup("tables");
  table.getMapKeys(keys);
  for (auto key : keys) {
    auto componentERT = table.lookup(key);
    auto pos = key.rfind(".");
    auto componentName = key.substr(pos + 1, key.size() - pos - 1);
    // std::cout << componentName << std::endl;

    // update levels by name and the type of it
    if (componentName == "wire" || componentName == "Wire") { // special case, update interal wire model
      float transferEnergy;
      auto actionERT = componentERT.lookup("transfer_random");
      if (actionERT.lookupValue("energy", transferEnergy)) {
        for (unsigned i = 0; i < NumStorageLevels(); i++) { // update wire energy for all storage levels
          auto bufferSpec = GetStorageLevel(i);
          auto networkSpec = GetNetwork(i); // FIXME.
          std::static_pointer_cast<LegacyNetwork::Specs>(networkSpec)->wire_energy = transferEnergy; // FIXME.
        }
      }
    } else {
      // Find the level that matches this name and see what type it is
      bool isArithmeticUnit = false;
      bool isBuffer = false;
      std::shared_ptr<LevelSpecs> specToUpdate;
      for (auto level : levels) {
        //std::cout << "  level: " << level->level_name << std::endl;
        if (level->level_name == componentName) {
          specToUpdate = level;
          if (level->Type() == "BufferLevel") isBuffer = true;
          if (level->Type() == "ArithmeticUnits") isArithmeticUnit = true;
        }
      }
      // Find the most expensive action as the unit cost
      std::vector<std::string> actions;
      componentERT.getMapKeys(actions);
      double opEnergy = 0.0;
      double argEnergy = 0.0;
      for (auto action : actions) {
        auto actionERT = componentERT.lookup(action);
        if (actionERT.isList()) { // action support argument
          for (int i = 0; i < actionERT.getLength(); i ++) {
            if (actionERT[i].lookupValue("energy", argEnergy)) {
              opEnergy = std::max(argEnergy, opEnergy);
            }
          }
        } else { // no argument action
          if (actionERT.lookupValue("energy", argEnergy)) {
            opEnergy = std::max(argEnergy, opEnergy);
          }
        }
      }
      // Replace the energy per action
      if (isArithmeticUnit) {
        // std::cout << "  Replace " << componentName << " energy with energy " << opEnergy << std::endl;
        auto arithmeticSpec = GetArithmeticLevel();
        arithmeticSpec->energy_per_op = opEnergy;
      } else if (isBuffer) {
        auto bufferSpec = std::static_pointer_cast<BufferLevel::Specs>(specToUpdate);
        // std::cout << "  Replace " << componentName << " VectorAccess energy with energy " << opEnergy << std::endl;
        bufferSpec->vector_access_energy = opEnergy / bufferSpec->cluster_size.Get();
      } else {
        // std::cout << "  Unused component ERT: "  << key << std::endl;
      }
    }
  }

  return;
}

std::vector<std::string> Topology::Specs::LevelNames() const
{
  std::vector<std::string> level_names;
  for (unsigned level_id = 0; level_id < NumLevels(); level_id++)
  {
    level_names.push_back(GetLevel(level_id)->level_name);
  }
  return level_names;
}

std::vector<std::string> Topology::Specs::StorageLevelNames() const
{
  std::vector<std::string> storage_level_names;
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    storage_level_names.push_back(GetStorageLevel(storage_level_id)->level_name);
  }
  return storage_level_names;
}

//
// Level accessors.
//

void Topology::Specs::AddLevel(unsigned typed_id, std::shared_ptr<LevelSpecs> level_specs)
{
  if (level_specs->Type() == "BufferLevel")
  {
    storage_map[typed_id] = levels.size();
  }
  else if (level_specs->Type() == "ArithmeticUnits")
  {
    assert(typed_id == 0);
    arithmetic_map = levels.size();
  }
  else
  {
    std::cerr << "ERROR: illegal level specs type: " << level_specs->Type() << std::endl;
    exit(1);
  }
  levels.push_back(level_specs);
}

void Topology::Specs::AddInferredNetwork(std::shared_ptr<LegacyNetwork::Specs> specs)
{
  inferred_networks.push_back(specs);
}

void Topology::Specs::AddNetwork(std::shared_ptr<NetworkSpecs> specs)
{
  networks.push_back(specs);
}

unsigned Topology::Specs::NumLevels() const
{
  return levels.size();
}

unsigned Topology::Specs::NumStorageLevels() const
{
  return storage_map.size();
}

unsigned Topology::Specs::NumNetworks() const
{
  return networks.size();
}

std::shared_ptr<LevelSpecs> Topology::Specs::GetLevel(unsigned level_id) const
{
  return levels.at(level_id);
}

std::shared_ptr<BufferLevel::Specs> Topology::Specs::GetStorageLevel(unsigned storage_level_id) const
{
  auto level_id = storage_map.at(storage_level_id);
  return std::static_pointer_cast<BufferLevel::Specs>(levels.at(level_id));
}

std::shared_ptr<ArithmeticUnits::Specs> Topology::Specs::GetArithmeticLevel() const
{
  auto level_id = arithmetic_map;
  return std::static_pointer_cast<ArithmeticUnits::Specs>(levels.at(level_id));
}

std::shared_ptr<LegacyNetwork::Specs> Topology::Specs::GetInferredNetwork(unsigned network_id) const
{
  return inferred_networks.at(network_id);
}

std::shared_ptr<NetworkSpecs> Topology::Specs::GetNetwork(unsigned network_id) const
{
  return networks.at(network_id);
}

//--------------------------------------------//
//                  Topology                  //
//--------------------------------------------//

std::ostream& operator << (std::ostream& out, const Topology& topology)
{
  // Save ios format state.
  std::ios state(NULL);
  state.copyfmt(out);
  out << std::fixed << std::setprecision(2);

  //
  // Detailed specs and stats.
  //

  out << "Buffer and Arithmetic Levels" << std::endl;
  out << "----------------------------" << std::endl;

  int level_id = 0;
  for (auto & level : topology.levels_)
  {
    out << "Level " << level_id << std::endl;
    out << "-------" << std::endl;
    out << *level;
    level_id++;
  }

  out << "Networks" << std::endl;
  out << "--------" << std::endl;

#define PRINT_NETWORKS_IN_LEGACY_ORDER
#ifdef PRINT_NETWORKS_IN_LEGACY_ORDER
  for (unsigned storage_level_id = 0; storage_level_id < topology.NumStorageLevels(); storage_level_id++)
  {
    auto network_id = storage_level_id;
    auto network = topology.GetStorageLevel(storage_level_id)->GetReadNetwork();
    out << "Network " << network_id << std::endl;
    out << "---------" << std::endl;
    out << *network;
  }
#else
  int network_id = 0;
  for (auto & network : topology.networks_)
  {
    out << "Network " << network_id << std::endl;
    out << "---------" << std::endl;
    out << *(network.second);
    network_id++;
  }  
#endif

  if (topology.is_evaluated_)
  {
    out << "Total topology energy: " << topology.stats_.energy << " pJ" << std::endl;
    out << "Total topology area: " << topology.stats_.area << " um^2" << std::endl;
    out << "Max topology cycles: " << topology.stats_.cycles << std::endl;
  }

  out << std::endl;

  //
  // Summary stats.
  //
  out << "Summary Stats" << std::endl;
  out << "-------------" << std::endl;
    
  if (topology.is_evaluated_)
  {
    out << "Utilization: " << topology.stats_.utilization << std::endl;
    out << "Cycles: " << topology.stats_.cycles << std::endl;
    out << "Sparse Cycles: " << topology.stats_.sparse_cycles << std::endl;
    out << "Energy: " << topology.stats_.energy / 1000000 << " uJ" << std::endl;
    out << "Sparse Energy: " << topology.stats_.sparse_energy / 1000000 << " uJ" << std::endl;
  }
  out << "Area: " << topology.stats_.area / 1000000 << " mm^2" << std::endl;
  out << std::endl;

  if (topology.is_evaluated_)
  {
    auto num_maccs = topology.stats_.maccs;
    out << "MACCs = " << num_maccs << std::endl;
    out << "pJ/MACC" << std::endl;

    std::size_t max_name_length = 0;
    for (unsigned i = 0; i < topology.NumLevels(); i++)
    {
      max_name_length = std::max(max_name_length, topology.GetLevel(i)->Name().length());
    }

    for (auto& network: topology.networks_)
    {
      max_name_length = std::max(max_name_length, network.second->Name().length());
    }

    std::string indent = "    ";
    int align = max_name_length + 1;

    if(topology.GetDenseSuccess())
    {
      out << "Dense MACCs = " << num_maccs << std::endl;
      out << "Dense pJ/MACC" << std::endl;

      for (unsigned i = 0; i < topology.NumLevels(); i++)
      {
        auto level = topology.GetLevel(i);
        out << "    " << std::setw(align) << std::left << level->Name() << "= "
            << level->Energy() / num_maccs << std::endl;
      }
      out << "    " << std::setw(align) << std::left << "Total" << "= " << topology.stats_.energy / num_maccs << std::endl;

      out << std::endl;
    }
    else
    {
      out << "Dense MACCs = " << num_maccs << std::endl;
      out << "Invalid Dense Mapping, Dense Working Set Does Not Fit the Buffer Constaints." << std::endl;
      out << std::endl;
    }

    out << "Sparse pJ / Dense MACC" << std::endl;

    for (unsigned i = 0; i < topology.NumLevels(); i++)
    {
      auto level = topology.GetLevel(i);
      out << indent << std::setw(align) << std::left << level->Name() << "= "
          << level->SparseEnergy() / num_maccs << std::endl;
    }
    out << indent << std::setw(align) << std::left << "Total" << "= "
        << topology.SparseEnergy() / num_maccs << std::endl;

    out << std::endl;

    out << "Speedup: " << topology.Cycles() / static_cast<double>(topology.SparseCycles()) <<"x"<<std::endl;
    
    if(topology.GetDenseSuccess())
      out << "Total Energy reduction: " << topology.Energy() / topology.SparseEnergy() <<"x"<<std::endl;
    else
      out << "Total Energy reduction unknown. "<<std::endl;

#ifdef PRINT_NETWORKS_IN_LEGACY_ORDER
    for (unsigned storage_level_id = 0; storage_level_id < topology.NumStorageLevels(); storage_level_id++)
    {
      auto network = topology.GetStorageLevel(storage_level_id)->GetReadNetwork();
      out << indent << std::setw(align) << std::left << network->Name() << "= "
          << network->Energy() / num_maccs << std::endl;
    }
#else
    for (auto& network: topology.networks_)
    {
      out << indent << std::setw(align) << std::left << network.second->Name() << "= "
          << network.second->Energy() / num_maccs << std::endl;
    }
#endif

    out << indent << std::setw(align) << std::left << "Total" << "= "
        << topology.Energy() / num_maccs << std::endl;
  }

  // Restore ios format state.
  out.copyfmt(state);

  return out;
}

void Topology::Spec(const Topology::Specs& specs)
{
  specs_ = specs;

  for (auto& level : levels_)
  {
    level.reset();
  }
  levels_.clear();

  for (auto& network : networks_)
  {
    network.second.reset();
  }
  networks_.clear();

  // Construct and spec Buffer and Arithmeic levels.
  for (unsigned i = 0; i < specs.NumLevels(); i++)
  {
    auto level_specs = specs.GetLevel(i);
    // auto level_specs = specs_.GetLevel(i); difference between the above ?
      
    // What type of level is this?
    if (level_specs->Type() == "BufferLevel")
    {
      BufferLevel::Specs& specs = *std::static_pointer_cast<BufferLevel::Specs>(level_specs);
      std::shared_ptr<BufferLevel> buffer_level = std::make_shared<BufferLevel>(specs); // creation of buffer level object
      std::shared_ptr<Level> level = std::static_pointer_cast<Level>(buffer_level);
      levels_.push_back(level);
    }
    else if (level_specs->Type() == "ArithmeticUnits")
    {
      ArithmeticUnits::Specs& specs = *std::static_pointer_cast<ArithmeticUnits::Specs>(level_specs);
      std::shared_ptr<ArithmeticUnits> arithmetic_level = std::make_shared<ArithmeticUnits>(specs);
      std::shared_ptr<Level> level = std::static_pointer_cast<Level>(arithmetic_level);
      levels_.push_back(level);
    }
    else
    {
      std::cerr << "ERROR: illegal level specs type: " << level_specs->Type() << std::endl;
      exit(1);
    }
  }

  //
  // Construct and spec user-defined networks. Auto-inferred networks will be
  // generated on-demand as we walk through the connection code.
  // 
  for (unsigned i = 0; i < specs.NumNetworks(); i++)
  {
    auto network = NetworkFactory::Construct(specs.GetNetwork(i));
    networks_[network->Name()] = network;
  }

  //
  // Connect levels to networks.
  // FIXME: network source and sink need to be *bound* per-dataspace at eval (mapping) time.
  //
  for (unsigned i = 0; i < specs.NumLevels()-1; i++)
  {
    // Note! We are linking levels[i+1] as the outer level for networks[i].
    auto inner = levels_.at(i);
    auto outer = levels_.at(i+1);

    bool inner_is_arithmetic = (i == 0);

    auto outer_buffer = std::static_pointer_cast<BufferLevel>(outer);

    std::shared_ptr<Network> read_fill_network = nullptr;
    std::shared_ptr<Network> drain_update_network = nullptr;

    //
    // Find read-fill network.
    //
    if (outer_buffer->GetSpecs().read_network_name.IsSpecified())
    {
      std::string outer_read_name = outer_buffer->GetSpecs().read_network_name.Get();
      auto it = networks_.find(outer_read_name);
      if (it == networks_.end())
      {
        std::cerr << "ERROR: network " << outer_read_name << " not found." << std::endl;
        exit(1);
      }
      read_fill_network = it->second;

      if (!inner_is_arithmetic)
      {
        auto inner_buffer = std::static_pointer_cast<BufferLevel>(inner);
        assert(inner_buffer->GetSpecs().fill_network_name.IsSpecified());
        std::string inner_fill_name = inner_buffer->GetSpecs().fill_network_name.Get();
        assert(outer_read_name == inner_fill_name);
      }
      else
      {
        auto inner_arithmetic = std::static_pointer_cast<ArithmeticUnits>(inner);
        assert(inner_arithmetic->GetSpecs().operand_network_name.IsSpecified());
        std::string inner_operand_name = inner_arithmetic->GetSpecs().operand_network_name.Get();
        assert(outer_read_name == inner_operand_name);
      }
    }
    else // outer read network name is not specified.
    {
      // Create a new Legacy-type read-fill network from the pre-parsed inferred network specs.

      // Inferred network i connects outer-level (i+1) to inner-level (i).
      // Inferred network i connects outer-storage-level (i) to inner-storage-level (i-1).
      auto inferred_network_id = i;
      auto inferred_network_specs = *specs.GetInferredNetwork(inferred_network_id);
      std::shared_ptr<LegacyNetwork> legacy_network = std::make_shared<LegacyNetwork>(inferred_network_specs);
      std::shared_ptr<Network> network = std::static_pointer_cast<Network>(legacy_network);

      std::string network_name = outer->Name() + " <==> " + inner->Name();
      network->SetName(network_name);

      networks_[network_name] = network;
      read_fill_network = network;

      if (!inner_is_arithmetic)
      {
        auto inner_buffer = std::static_pointer_cast<BufferLevel>(inner);
        assert(!inner_buffer->GetSpecs().fill_network_name.IsSpecified());          
      }
      else
      {
        auto inner_arithmetic = std::static_pointer_cast<ArithmeticUnits>(inner);
        assert(!inner_arithmetic->GetSpecs().operand_network_name.IsSpecified());          
      }
    }

    //
    // Find drain-update network.
    //
    if (outer_buffer->GetSpecs().update_network_name.IsSpecified())
    {
      std::string outer_update_name = outer_buffer->GetSpecs().update_network_name.Get();
      auto it = networks_.find(outer_update_name);
      if (it == networks_.end())
      {
        std::cerr << "ERROR: network " << outer_update_name << " not found." << std::endl;
        exit(1);
      }
      drain_update_network = it->second;

      if (!inner_is_arithmetic)
      {
        auto inner_buffer = std::static_pointer_cast<BufferLevel>(inner);
        assert(inner_buffer->GetSpecs().drain_network_name.IsSpecified());
        std::string inner_drain_name = inner_buffer->GetSpecs().drain_network_name.Get();
        assert(outer_update_name == inner_drain_name);
      }
      else
      {
        auto inner_arithmetic = std::static_pointer_cast<ArithmeticUnits>(inner);
        assert(inner_arithmetic->GetSpecs().result_network_name.IsSpecified());
        std::string inner_result_name = inner_arithmetic->GetSpecs().result_network_name.Get();
        assert(outer_update_name == inner_result_name);
      }
    }
    else // outer update network name is not specified.
    {
      // Reuse the existing read-fill network.
      assert(read_fill_network != nullptr);
      drain_update_network = read_fill_network;

      if (!inner_is_arithmetic)
      {
        auto inner_buffer = std::static_pointer_cast<BufferLevel>(inner);
        assert(!inner_buffer->GetSpecs().drain_network_name.IsSpecified());
      }
      else
      {
        auto inner_arithmetic = std::static_pointer_cast<ArithmeticUnits>(inner);
        assert(!inner_arithmetic->GetSpecs().result_network_name.IsSpecified());
      }
    }

    //
    // We've found the network objects, now make the connections.
    //

    outer_buffer->ConnectRead(read_fill_network);
    outer_buffer->ConnectUpdate(drain_update_network);

    if (!inner_is_arithmetic)
    {
      auto inner_buffer = std::static_pointer_cast<BufferLevel>(inner);
      inner_buffer->ConnectFill(read_fill_network);
      inner_buffer->ConnectDrain(drain_update_network);
    }
    else
    {
      auto inner_arithmetic = std::static_pointer_cast<ArithmeticUnits>(inner);
      inner_arithmetic->ConnectOperand(read_fill_network);
      inner_arithmetic->ConnectResult(drain_update_network);
    }

    // If the same bi-directional network is used for read-fill and drain-update,
    // set the source/sink links for the drain-update direction. This is because
    // the network queries the sink's read-modify-write ability to determine whether
    // read-modify-write traffic incurs double the number of transfers.
    // FIXME:
    // (1) Come up with a better name than "source" and "sink" for bidirectional
    //     networks.
    // (2) Perhaps the read-modify-write traffic scaling should be performed outside
    //     of the network model (and be part of the higher-level tile analysis).
    drain_update_network->ConnectSource(inner);
    drain_update_network->ConnectSink(outer);

    if (drain_update_network != read_fill_network)
    {
      read_fill_network->ConnectSource(outer);
      read_fill_network->ConnectSink(inner);
    }
  } // for all levels.

  // DeriveFanouts();

  is_specced_ = true;

  FloorPlan();

  // Compute area at spec-time (instead of at eval-time).
  // FIXME: area is being stored as a stat here, while it is stored as a spec
  // in individual modules. We need to be consistent.
  double area = 0;
  for (auto level : levels_)
  {
    assert(level->Area() >= 0);
    area += level->Area();
  }
  stats_.area = area;
}

// The hierarchical ParseSpecs functions are static and do not
// affect the internal specs_ data structure, which is set by
// the dynamic Spec() call later.

// This function implements the "classic" hierarchical topology
// with arithmetic units at level 0 and storage units at level 1+.
Topology::Specs Topology::ParseSpecs(config::CompoundConfigNode storage,
                                     config::CompoundConfigNode arithmetic)
{
  Specs specs;
  
  assert(storage.isList());

  // Level 0: arithmetic.
  // Use multiplication factor == 0 to ensure .instances attribute is set
  auto level_specs_p = std::make_shared<ArithmeticUnits::Specs>(ArithmeticUnits::ParseSpecs(arithmetic, 0));
  specs.AddLevel(0, std::static_pointer_cast<LevelSpecs>(level_specs_p));

  // Storage levels.
  int num_storage_levels = storage.getLength();
  for (int i = 0; i < num_storage_levels; i++)
  {
    auto level_specs_p = std::make_shared<BufferLevel::Specs>(BufferLevel::ParseSpecs(storage[i], 0));
    specs.AddLevel(i, std::static_pointer_cast<LevelSpecs>(level_specs_p));

    if(i>0) // connect previous level's next to current level
      specs.GetStorageLevel(i-1)->next = level_specs_p;

    // For each storage level, parse and extract an inferred network spec from the storage config.
    // A network object corresponding to this spec will only be instantiated if a user-specified
    // network is missing between any two topology levels.
    auto inferred_network_specs_p = std::make_shared<LegacyNetwork::Specs>(LegacyNetwork::ParseSpecs(storage[i], 0));
    specs.AddInferredNetwork(inferred_network_specs_p);
  }

  return specs;
}

// This function implements the "tree-like" hierarchical architecture description
// used in Accelergy v0.2. The lowest level is level 0 and should have
// arithmetic units, while other level are level 1+ with some buffer/storage units
Topology::Specs Topology::ParseTreeSpecs(config::CompoundConfigNode designRoot)
{
  Specs specs;
  auto curNode = designRoot;

  std::vector<std::shared_ptr<LevelSpecs>> storages; // serialize all storages
  std::vector<std::shared_ptr<LegacyNetwork::Specs>> inferred_networks;
  std::vector<std::shared_ptr<NetworkSpecs>> networks;

  uint32_t multiplication = 1;

  // Walk the tree to find each buffer and arithmetic units
  // and add them to the specs.
  while (curNode.exists("subtree"))
  {
    auto subTrees = curNode.lookup("subtree");
    // Timeloop currently supports one subtree per level.
    assert(subTrees.isList() && subTrees.getLength() == 1);
    curNode = subTrees[0];

    std::string curNodeName;
    curNode.lookupValue("name", curNodeName);

    uint32_t subTreeSize = config::parseElementSize(curNodeName);
    multiplication *= subTreeSize;

    if (curNode.exists("local"))
    {
      auto curLocal = curNode.lookup("local");
      assert(curLocal.isList());

      std::vector<std::shared_ptr<LevelSpecs>> localStorages;
      std::vector<std::shared_ptr<LegacyNetwork::Specs>> localInferredNetworks;
      std::vector<std::shared_ptr<NetworkSpecs>> localNetworks;

      for (int c = 0; c < curLocal.getLength() ; c++)
      {
        std::string cName, cClass;
        curLocal[c].lookupValue("name", cName);
        curLocal[c].lookupValue("class", cClass);
        uint32_t localElementSize = config::parseElementSize(cName);
        uint32_t nElements = multiplication * localElementSize;

        if (isBufferClass(cClass))
        {
          // Create a buffer spec.
          auto level_specs_p = std::make_shared<BufferLevel::Specs>(BufferLevel::ParseSpecs(curLocal[c], nElements));
          localStorages.push_back(level_specs_p);

          // Create an inferred network spec.
          // A network object corresponding to this spec will only be instantiated if a user-specified
          // network is missing between any two topology levels.
          auto inferred_network_specs_p = std::make_shared<LegacyNetwork::Specs>(LegacyNetwork::ParseSpecs(curLocal[c], nElements));
          localInferredNetworks.push_back(inferred_network_specs_p);
        }
        else if (isComputeClass(cClass))
        {
          // Create arithmetic.
          auto level_specs_p = std::make_shared<ArithmeticUnits::Specs>(ArithmeticUnits::ParseSpecs(curLocal[c], nElements));
          specs.AddLevel(0, std::static_pointer_cast<LevelSpecs>(level_specs_p));
        }
        else if (isNetworkClass(cClass))
        {
          auto network_specs_p = NetworkFactory::ParseSpecs(curLocal[c], nElements);
          localNetworks.push_back(network_specs_p);
        }
        else
        {
          // std::cout << "  Neglect component: " << cName << " due to unknown class: " << cClass << std::endl;
        }
      }
      // The deeper the tree, the closer the buffer to be with ArithmeticUnits.
      // Reverse the order so that top in the local list is at the bottem, matching the tree seq
      storages.insert(storages.begin(), localStorages.rbegin(), localStorages.rend());
      inferred_networks.insert(inferred_networks.begin(), localInferredNetworks.rbegin(), localInferredNetworks.rend());
      networks.insert(networks.begin(), localNetworks.rbegin(), localNetworks.rend());
    }
  } // end while

  // Add storages to specs. We can do this only after walking the whole tree.
  for (uint32_t i = 0; i < storages.size(); i++)
  {
    auto storage = storages[i];

    std::cout << "AddLevel (storage) : " << i << " " << storage->level_name << std::endl;
    if(i>0) // connect previous level's next to current level
      specs.GetStorageLevel(i-1)->next = std::static_pointer_cast<BufferLevel::Specs>(storage);

    specs.AddLevel(i, storage);

    auto inferred_network = inferred_networks[i];
    specs.AddInferredNetwork(inferred_network);
  }

  // Add user-specified networks.
  for (unsigned i = 0; i < networks.size(); i++)
  {
    auto network = networks[i];
    specs.AddNetwork(network);
  }

  return specs;
};

unsigned Topology::NumLevels() const
{
  assert(is_specced_);
  return levels_.size();
}

unsigned Topology::NumStorageLevels() const
{
  assert(is_specced_);
  return specs_.NumStorageLevels();
}

unsigned Topology::NumNetworks() const
{
  assert(is_specced_);
  return specs_.NumNetworks();
}

std::shared_ptr<Level> Topology::GetLevel(unsigned level_id) const
{
  return levels_.at(level_id);
}

std::shared_ptr<BufferLevel> Topology::GetStorageLevel(unsigned storage_level_id) const
{
  auto level_id = specs_.StorageMap(storage_level_id);
  return std::static_pointer_cast<BufferLevel>(levels_.at(level_id));
}

std::shared_ptr<ArithmeticUnits> Topology::GetArithmeticLevel() const
{
  auto level_id = specs_.ArithmeticMap();
  return std::static_pointer_cast<ArithmeticUnits>(levels_.at(level_id));
}

// PreEvaluationCheck(): allows for a very fast capacity-check
// based on given working-set sizes that can be trivially derived
// by the caller. The more powerful Evaluate() function also
// performs these checks, but computes both tile sizes and access counts
// and requires full tiling data that is generated by a very slow
// Nest::ComputeWorkingSets() algorithm. The PreEvaluationCheck()
// function is an optional call that extensive design-space searches
// can use to fail early.
// FIXME: integrate with Evaluate() and re-factor.
// FIXME: what about instances and fanout checks?
std::vector<EvalStatus> Topology::PreEvaluationCheck(const Mapping& mapping,
                                                     analysis::NestAnalysis* analysis,
                                                     bool break_on_failure)
{
  auto masks = tiling::TransposeMasks(mapping.datatype_bypass_nest);
  auto working_set_sizes = analysis->GetWorkingSetSizes_LTW();
  // get max or average sparsity
  // auto sparsity = analysis->GetMaximumSparsity(); // most dense sample
  auto sparsity = analysis->GetAverageSparsity(); // allow as long as average sample density can fit

  // allow evaluation as long as max or average sparse working set can fit in the buffer
  auto wid = problem::GetShape()->name == "Weight-Update-Depthwise" ? 
                  (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : 
                  (problem::GetShape()->DataSpaceNameToID.at("Weights"));
  for(auto &level_ws: working_set_sizes)
    level_ws[wid] = std::ceil(sparsity * level_ws[wid]);

  auto num_storage_levels = working_set_sizes.size();
  assert(num_storage_levels == NumStorageLevels());
  available_weight_capacity_budget_.resize(num_storage_levels, 0);
  std::vector<EvalStatus> eval_status(NumLevels(), { .success = true, .fail_reason = "" });
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    auto level_id = specs_.StorageMap(storage_level_id);
    auto s = GetStorageLevel(storage_level_id)->PreEvaluationCheck( // pass into a reference for buffer size
      working_set_sizes.at(storage_level_id), 
      masks.at(storage_level_id),
      break_on_failure);

    if(s.success && masks.at(storage_level_id).at(wid) && storage_level_id != NumStorageLevels()-1)
    {
      available_weight_capacity_budget_.at(storage_level_id) = GetStorageLevel(storage_level_id)->AvailableBudget(
        working_set_sizes.at(storage_level_id), masks.at(storage_level_id));
      assert(available_weight_capacity_budget_.at(storage_level_id)>0);
    }
    eval_status.at(level_id) = s;
    if (break_on_failure && !s.success)
      break;
  }

  return eval_status;
}

std::vector<EvalStatus> Topology::Evaluate(Mapping& mapping,
                                           analysis::NestAnalysis* analysis,
                                           const problem::Workload& workload,
                                           bool break_on_failure)
{
  assert(is_specced_);

  // ==================================================================
  // TODO: connect buffers to networks based on bypass mask in mapping.
  // ==================================================================
  // for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  // {
  //   auto storage_level = GetStorageLevel(storage_level_id);
  //   auto network = GetNetwork(storage_level_id);

  //   storage_level->ConnectNetwork(network);
  //   network->ConnectBuffer(storage_level);
  // }  

  std::vector<EvalStatus> eval_status(NumLevels(), { .success = true, .fail_reason = "" });
  bool success_accum = true;
  
  // Compute working-set tile hierarchy for the nest.
  problem::PerDataSpace<std::vector<tiling::TileInfo>> ws_tiles;
  try
  {
    ws_tiles = analysis->GetWorkingSets(available_weight_capacity_budget_);
  }
  catch (std::runtime_error& e)
  {
    std::fill(eval_status.begin(), eval_status.end(),
              EvalStatus({ .success = false, .fail_reason = "" }));
    return eval_status;
  }

  // Ugh... FIXME.
  auto compute_cycles = analysis->GetBodyInfo().accesses;

  // sparsity info
  const auto &sparsity_info = analysis->GetSparsityInfo();

  // Create a mask indicating which levels support distributed multicast.
  tiling::CompoundMaskNest distribution_supported;
  for (unsigned pv = 0; pv < unsigned(problem::GetShape()->NumDataSpaces); pv++)
  {
    distribution_supported[pv].reset();
    for (unsigned storage_level = 0; storage_level < NumStorageLevels(); storage_level++)
    {
      if (GetStorageLevel(storage_level)->GetReadNetwork()->DistributedMulticastSupported())
      {
        distribution_supported[pv].set(storage_level);
      }
    }
  }
  
  // Collapse tiles into a specified number of tiling levels. The solutions are
  // received in a set of per-problem::Shape::DataSpaceID arrays.
  auto collapsed_tiles = tiling::CollapseTiles(ws_tiles, specs_.NumStorageLevels(),
                                               mapping.datatype_bypass_nest,
                                               distribution_supported);

  // Transpose the tiles into level->datatype structure.
  auto tiles = tiling::TransposeTiles(collapsed_tiles);
  assert(tiles.size() == NumStorageLevels());

  // Transpose the datatype bypass nest into level->datatype structure.
  auto keep_masks = tiling::TransposeMasks(mapping.datatype_bypass_nest);
  assert(keep_masks.size() >= NumStorageLevels());

  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    auto storage_level = GetStorageLevel(storage_level_id);
    
    // Evaluate Loop Nest on hardware structures: calculate
    // primary statistics.
    auto level_id = specs_.StorageMap(storage_level_id);

    // currently only sparse mapping are supported, 
    // we can obtain best dense mapping from original timeloop
    // This is currently not critical, dingqing FIXME later
    assert(gSparseEvaluate); 
    
    EvalStatus s;
    if(!gSparseEvaluate) // Dense
    {
      s = storage_level->Evaluate(tiles[storage_level_id], keep_masks[storage_level_id],
                                       compute_cycles, break_on_failure);
    }
    else // Sparse
    {
      // overflow evalutation is now outside the main SparseEvaluate function
      // since overflow require to use multilevel information which is not present in SparseEvaluate
      // We only need to evaluate Overflow for target levels, and for non-target level, we have a 
      // sanity check to make sure overflow is properly set to zero.
      
      // clear the overflow field as no new buffer object will be instantiated across different searched mappings.
      storage_level->ResetOverflow(); // how to make it more elegant?
      bool is_target_level = (sparsity_info.overflow_breakdown[storage_level_id].size() > 0);
      if(is_target_level)
      {
        storage_level->ComputeOverflow(storage_level_id, 
                                      sparsity_info.overflow_breakdown[storage_level_id], 
                                      mapping.datatype_bypass_nest);
      }
      else
        storage_level->CheckNoOverflow();

      // dingqing FIXME: glb level is not necessarily 1 in general! OK for original Eyeriss-like architecture but not general
      unsigned long sorting_blk_ptr_access = (storage_level_id==1)?sparsity_info.blk_ptr_access:0;
      
      s = storage_level->SparseEvaluate(tiles[storage_level_id], keep_masks[storage_level_id],
                                       compute_cycles, break_on_failure, 
                                       sparsity_info.weight_sparsity, /* sparsity_info.overflow_breakdown[storage_level_id],*/
                                       sparsity_info.unfilled[storage_level_id],
                                       mapping, !static_cast<bool>(storage_level_id), // pass mapping for debug
                                       workload, sparsity_info.inverse_speedup, 
                                       sorting_blk_ptr_access, sparsity_info.subtile); 
    }

    eval_status.at(level_id) = s;
    success_accum &= s.success;
    dense_success_ &= storage_level->GetDenseSuccess();

    if (break_on_failure && !s.success)
      break;

    // Evaluate network.
    // FIXME: move this out of this loop.    
    auto network = storage_level->GetReadNetwork(); // GetNetwork(storage_level_id);
    s = network->Evaluate(tiles[storage_level_id], break_on_failure);
    eval_status.at(level_id) = s;
    success_accum &= s.success;

    if (break_on_failure && !s.success)
      break;    
  }

  if (!break_on_failure || success_accum)
  {
    auto level_id = specs_.ArithmeticMap();
    auto s = GetArithmeticLevel()->HackEvaluate(analysis, workload);
    eval_status.at(level_id) = s;
    success_accum &= s.success;
  }

  if (!break_on_failure || success_accum)
  {
    ComputeStats();
  }

  if (success_accum)
  {
    is_evaluated_ = true;
  }

  return eval_status;
}

void Topology::ComputeStats()
{
  // Energy.
  double energy = 0;
  double sparse_energy = 0;
  for (auto level : levels_)
  {
    assert(level->Energy() >= 0);
    assert(level->SparseEnergy() >= 0);
    energy += level->Energy();
    sparse_energy += level->SparseEnergy();
  }

  for (auto& network: networks_)
  {
    //poan: users might add a network to the arch but never connect/use it
    if (!network.second->IsEvaluated()) continue;
    auto e = network.second->Energy();
    assert(e >= 0);
    energy += e;
    sparse_energy += e;
  }

  stats_.energy = energy;
  stats_.sparse_energy = sparse_energy;

  // Cycles.
  std::uint64_t cycles = 0;
  std::uint64_t sparse_cycles = 0;
  for (auto level : levels_)
  {
    cycles = std::max(cycles, level->Cycles());
    sparse_cycles = std::max(sparse_cycles, level->SparseCycles());
  }
  stats_.cycles = cycles;
  stats_.sparse_cycles = sparse_cycles;

  // Utilization.
  // FIXME.
  stats_.utilization = GetArithmeticLevel()->IdealCycles() / stats_.cycles;

  // Tile sizes.
  stats_.tile_sizes.clear();
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    problem::PerDataSpace<std::uint64_t> ts;
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      ts[pv] = GetStorageLevel(storage_level_id)->UtilizedCapacity(pv);
    }
    stats_.tile_sizes.push_back(ts);
  }

  // Utilized instances.
  stats_.utilized_instances.clear();
  for (unsigned storage_level_id = 0; storage_level_id < NumStorageLevels(); storage_level_id++)
  {
    problem::PerDataSpace<std::uint64_t> uc;
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      auto pv = problem::Shape::DataSpaceID(pvi);
      uc[pv] = GetStorageLevel(storage_level_id)->UtilizedInstances(pv);
    }
    stats_.utilized_instances.push_back(uc);
  }

  // MACCs.
  stats_.maccs = GetArithmeticLevel()->MACCs();

  // Last-level accesses.
  stats_.last_level_accesses = GetStorageLevel(NumStorageLevels()-1)->Accesses();
}

//
// Floorplanner.
//
void Topology::FloorPlan()
{
  // Area of all the compute + buffer elements in inner levels
  // (needed for wire energy calculation).
  double cur_tile_area = 0;
  std::uint64_t inner_instances = 0;

  // Compute the area of each hierarchical tile.
  for (unsigned i = 0; i < NumLevels(); i++)
  {
    unsigned fanout;
    std::uint64_t cur_instances;
    if (i == 0)
    {
      cur_instances = GetArithmeticLevel()->GetSpecs().instances.Get();
      fanout = 0;
    }
    else
    {
      cur_instances = GetStorageLevel(i-1)->GetSpecs().instances.Get();
      assert(inner_instances % cur_instances == 0);
      fanout  = inner_instances / cur_instances;
    }
    inner_instances = cur_instances;

    cur_tile_area = GetLevel(i)->AreaPerInstance() + (cur_tile_area * fanout);
    tile_area_[i] = cur_tile_area;
  }

  // Tell each network the width of each hop's tile.
  for (unsigned i = 1; i < NumLevels(); i++)
  {
    auto level = GetLevel(i);
    auto storage_level = std::static_pointer_cast<BufferLevel>(level);
    auto network = storage_level->GetReadNetwork();
    double inner_tile_width = sqrt(tile_area_.at(i-1));
    network->SetTileWidth(inner_tile_width);
  }  
}

bool isBufferClass(std::string className)
{
  for (auto s : bufferClasses)
  {
    if (className.find(s) != std::string::npos) return true;
  }
  return false;
}

bool isComputeClass(std::string className)
{
  for (auto s : computeClasses)
  {
    if (className.find(s) != std::string::npos) return true;
  }
  return false;
}

bool isNetworkClass(std::string className)
{
  for (auto s : networkClasses)
  {
    if (className.find(s) != std::string::npos) return true;
  }
  return false;
}

}  // namespace model
