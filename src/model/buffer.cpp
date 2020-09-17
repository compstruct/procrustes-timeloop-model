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
#include <numeric>
#include <string>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "model/buffer.hpp"
//BOOST_CLASS_EXPORT(model::BufferLevel::Specs)
BOOST_CLASS_EXPORT(model::BufferLevel)

#include "util/numeric.hpp"
#include "util/misc.hpp"
#include "pat/pat.hpp"

bool gInfOverflowBuffer = false; // if true, we obtain a theoretical energy saving: all overflows are resolved in the current level
bool gEquivlentWU = (getenv("TIMELOOP_EQUIVLENT_WU") != NULL);

namespace model
{

// ==================================== //
//             Buffer Level             //
// ==================================== //

BufferLevel::BufferLevel()
{ }

BufferLevel::BufferLevel(const Specs& specs) :
    specs_(specs)
{
  is_specced_ = true;
  is_evaluated_ = false;
  overflow_access_ = 0.0;
  converted_overflow_ = 0.0;
}

BufferLevel::~BufferLevel()
{ }

// The hierarchical ParseSpecs functions are static and do not
// affect the internal specs_ data structure, which is set by
// the dynamic Spec() call later.
BufferLevel::Specs BufferLevel::ParseSpecs(config::CompoundConfigNode level, uint32_t n_elements)
{
  auto& buffer = level;

  Specs specs;

  // Name. This has to go first. Since the rest can be attributes
  std::string name;
  if (buffer.lookupValue("name", name))
  {
    specs.name = config::parseName(name);
  }

  std::string className = "";
  if (buffer.exists("attributes"))
  {
    buffer.lookupValue("class", className);
    buffer = buffer.lookup("attributes");
  }

  // Word Bits.
  std::uint32_t word_bits;
  if (buffer.lookupValue("word-bits", word_bits) ||
      buffer.lookupValue("word_width", word_bits) ||
      buffer.lookupValue("datawidth", word_bits) )
  {
    specs.word_bits = word_bits;
  }
  else
  {
    specs.word_bits = Specs::kDefaultWordBits;
  }

  // Block size.
  std::uint32_t block_size;
  specs.block_size = 1;
  if (buffer.lookupValue("block-size", block_size) ||
      buffer.lookupValue("n_words", block_size) )
  {
    specs.block_size = block_size;
  }

  // Cluster size.
  std::uint32_t cluster_size;
  specs.cluster_size = 1;
  std::uint32_t width;
  if (buffer.lookupValue("cluster-size", cluster_size))
  {
    specs.cluster_size = cluster_size;
  }
  else if (buffer.lookupValue("width", width))
  {
    word_bits = specs.word_bits.Get();
    block_size = specs.block_size.Get();
    assert(width % (word_bits * block_size)  == 0);
    specs.cluster_size = width / (word_bits * block_size);
  }

  // Size.
  // It has dependency on BlockSize and thus is initialized after BlockSize.
  std::uint32_t size;
  if (buffer.lookupValue("entries", size) )
  {
    assert(buffer.exists("sizeKB") == false);
    specs.size = size;
  }
  else if (buffer.lookupValue("depth", size) ||
           buffer.lookupValue("memory_depth", size))
  {
    assert(buffer.exists("sizeKB") == false);
    assert(buffer.exists("entries") == false);
    specs.size = size * specs.block_size.Get();
  }
  else if (buffer.lookupValue("sizeKB", size))
  {
    specs.size = size * 1024 * 8 / specs.word_bits.Get();
  }


  // Technology.
  // Unfortunately ".technology" means different things between ISPASS format
  // and Accelergy v0.2 format. So we use the class name to find out what to
  // assume.
  std::string technology;
  specs.technology = Technology::SRAM;
  if (className == "DRAM") specs.technology = Technology::DRAM;

  if (buffer.lookupValue("technology", technology) && technology == "DRAM")
  {
    specs.technology = Technology::DRAM;
  }

  // SRAM Type.
  std::uint32_t num_ports = 2;
  specs.num_ports = num_ports;
  if (buffer.lookupValue("num-ports", num_ports))
  {
    if (num_ports == 1)
    {
      specs.num_ports = num_ports;
    }
    else
    {
      assert(num_ports == 2);
    }
  }

  // Number of Banks.
  std::uint32_t num_banks = 2;
  specs.num_banks = num_banks;
  if (buffer.lookupValue("num-banks", num_banks))
  {
    specs.num_banks = num_banks;
  }

  // Bandwidth.
  double bandwidth;
  if (buffer.lookupValue("bandwidth", bandwidth))
  {
    std::cerr << "WARNING: bandwidth is deprecated. Assuming read_bandwidth = write_bandwidth = bandwidth/2" << std::endl;
    specs.read_bandwidth  = bandwidth / 2;
    specs.write_bandwidth = bandwidth / 2;
  }

  double read_bandwidth;
  if (buffer.lookupValue("read_bandwidth", read_bandwidth))
  {
    specs.read_bandwidth = read_bandwidth;
  }

  double write_bandwidth;
  if (buffer.lookupValue("write_bandwidth", write_bandwidth))
  {
    specs.write_bandwidth = write_bandwidth;
  }

  // Multiple-buffering factor (e.g., 2.0 means double buffering)
  double multiple_buffering;
  if (buffer.lookupValue("multiple-buffering", multiple_buffering))
  {
    specs.multiple_buffering = multiple_buffering;
  }
  else
  {
    specs.multiple_buffering = 1.0;
  }
  
  if (specs.size.IsSpecified())
  {
    specs.effective_size = static_cast<uint64_t>(std::floor(
            specs.size.Get() / specs.multiple_buffering.Get()));
  }

  // Minimum utilization factor (e.g., 1.0 requires full utilization of effective capacity)
  double min_utilizaiton;
  if (buffer.lookupValue("min-utilization", min_utilizaiton))
  {
    specs.min_utilization = min_utilizaiton;
  }
  else
  {
    specs.min_utilization = 0.0;
  }
  if (specs.min_utilization.Get() != 0.0)
  {
    assert(specs.effective_size.IsSpecified());
  }

  // Instances.
  std::uint32_t instances;
  if (buffer.lookupValue("instances", instances))
  {
    specs.instances = instances;
  } else {
    specs.instances = n_elements;
  }

  // MeshX.
  std::uint32_t meshX;
  if (buffer.lookupValue("meshX", meshX))
  {
    specs.meshX = meshX;
  }

  // MeshY.
  std::uint32_t meshY;
  if (buffer.lookupValue("meshY", meshY))
  {
    specs.meshY = meshY;
  }

  // Network names;
  std::string read_network_name;
  if (buffer.lookupValue("network_read", read_network_name))
  {
    specs.read_network_name = read_network_name;
  }

  std::string fill_network_name;
  if (buffer.lookupValue("network_fill", fill_network_name))
  {
    specs.fill_network_name = fill_network_name;
  }

  std::string drain_network_name;
  if (buffer.lookupValue("network_drain", drain_network_name))
  {
    specs.drain_network_name = drain_network_name;
  }

  std::string update_network_name;
  if (buffer.lookupValue("network_update", update_network_name))
  {
    specs.update_network_name = update_network_name;
  }

  // Vector Access Energy
  double tmp_access_energy = 0;
  double tmp_storage_area = 0;

  if (specs.technology.Get() == Technology::DRAM)
  {
    assert(specs.cluster_size.Get() == 1);
    tmp_access_energy = pat::DRAMEnergy(specs.word_bits.Get() * specs.block_size.Get());
    tmp_storage_area = 0;
  }
  else if (specs.size.Get() == 0)
  {
    //SRAM
    tmp_access_energy = 0;
    tmp_storage_area = 0;
  }
  else
  {
    std::uint64_t tmp_entries = specs.size.Get();
    std::uint64_t tmp_word_bits = specs.word_bits.Get();
    std::uint64_t tmp_block_size = specs.block_size.Get();
    std::uint64_t tmp_cluster_size = specs.cluster_size.Get();
    std::uint64_t width = tmp_word_bits * tmp_block_size * tmp_cluster_size;
    std::uint64_t height =
      (tmp_entries % tmp_block_size == 0) ?
      (tmp_entries / tmp_block_size)      :
      (tmp_entries / tmp_block_size) + 1;  

    tmp_access_energy = pat::SRAMEnergy(height, width, specs.num_banks.Get(), specs.num_ports.Get()) / tmp_cluster_size;
    tmp_storage_area = pat::SRAMArea(height, width, specs.num_banks.Get(), specs.num_ports.Get()) / tmp_cluster_size;
    std::cout << "Entries = " << tmp_entries
              << ", word_size = " << tmp_word_bits
              << ", block_size = " << tmp_block_size
              << ", cluster_size = " << tmp_cluster_size
              << ", num_banks = " << specs.num_banks.Get()
              << ", num_ports = " << specs.num_ports.Get()
              << ", energy = " << tmp_access_energy
              << ", area = " << tmp_storage_area << std::endl;
  }

  // Allow user to override the access energy.
  buffer.lookupValue("vector-access-energy", tmp_access_energy);

  // Allow user to override the cluster area.
  double tmp_cluster_area = 0;
  buffer.lookupValue("cluster-area", tmp_cluster_area);
  if (tmp_cluster_area > 0)
    tmp_storage_area = tmp_cluster_area / specs.cluster_size.Get();

  // Set final physical dimensions and energy.
  specs.vector_access_energy = tmp_access_energy;
  specs.storage_area = tmp_storage_area; //FIXME: check with Angshu

  std::cout << "BUFFER " << specs.name << " vector access energy = "
            << specs.vector_access_energy << " pJ, cluster area = "
            << specs.storage_area.Get() * specs.cluster_size.Get()
            << " um^2" << std::endl;

  specs.level_name = specs.name.Get();

  ValidateTopology(specs); // validate or infer instances == meshX * meshY
    
  return specs;
}

// Make sure the topology is consistent,
// and update unspecified parameters if they can
// be inferred from other specified parameters.
void BufferLevel::ValidateTopology(BufferLevel::Specs& specs)
{
  bool error = false;
  if (specs.instances.IsSpecified())
  {
    if (specs.meshX.IsSpecified())
    {
      if (specs.meshY.IsSpecified())
      {
        // All 3 are specified.
        assert(specs.meshX.Get() * specs.meshY.Get() == specs.instances.Get());
      }
      else
      {
        // Instances and MeshX are specified.
        assert(specs.instances.Get() % specs.meshX.Get() == 0);
        specs.meshY = specs.instances.Get() / specs.meshX.Get();
      }
    }
    else if (specs.meshY.IsSpecified())
    {
      // Instances and MeshY are specified.
      assert(specs.instances.Get() % specs.meshY.Get() == 0);
      specs.meshX = specs.instances.Get() / specs.meshY.Get();
    }
    else
    {
      // Only Instances is specified.
      specs.meshX = specs.instances.Get();
      specs.meshY = 1;
    }
  }
  else if (specs.meshX.IsSpecified())
  {
    if (specs.meshY.IsSpecified())
    {
      // MeshX and MeshY are specified.
      specs.instances = specs.meshX.Get() * specs.meshY.Get();
    }
    else
    {
      // Only MeshX is specified. We can make assumptions but it's too dangerous.
      error = true;
    }
  }
  else if (specs.meshY.IsSpecified())
  {
    // Only MeshY is specified. We can make assumptions but it's too dangerous.
    error = true;
  }
  else
  {
    // Nothing is specified.
    error = true;
  }

  if (error)
  {
    std::cerr << "ERROR: " << specs.name.Get()
              << ": instances and/or meshX * meshY must be specified."
              << std::endl;
    exit(1);        
  }
}

std::size_t BufferLevel::AvailableBudget(
    const problem::PerDataSpace<std::size_t> working_set_sizes,
    const tiling::CompoundMask mask)
{
  bool success = true;
  std::size_t available_weight_budget = 0;

  // Note that size for DRAM is usually not specified (assume inf size)
  // So size is not always specified
  assert(specs_.size.IsSpecified()); 
  if (specs_.size.IsSpecified())
  {
    auto available_capacity = specs_.effective_size.Get();
    if (network_read_->DistributedMulticastSupported())
    {
      available_capacity *= specs_.instances.Get();
    }

    // Find the total capacity required by all un-masked data types.
    std::size_t required_capacity = 0;

    // get id for sparse tensor
    // use weight for fw/bw and iact for wu
    // the exception is that FC/CONV wu is converted in python script
    // so look for iact for depthwise CONV, and every other case weight
    auto wid = problem::GetShape()->name == "Weight-Update-Depthwise" ? 
                  (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : 
                  (problem::GetShape()->DataSpaceNameToID.at("Weights")); // use this for available budget

    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      if (mask[pvi])
      {
        required_capacity += working_set_sizes.at(problem::Shape::DataSpaceID(pvi));
      }
    }

    if (required_capacity > available_capacity)
    {
      // std::cerr << "CAPACITY FAIL " << specs_.level_name << " req = " << required_capacity << " avail = " << available_capacity << std::endl;
      success = false;
    }
    else if (required_capacity < specs_.effective_size.Get()
             * specs_.min_utilization.Get())
    {
      success = false;
    }
    assert(success);
    assert(mask[wid]);

    if(success)
    {
      if(mask[wid])
      {
        available_weight_budget = available_capacity - (required_capacity - working_set_sizes.at(problem::Shape::DataSpaceID(wid)));
        assert(available_weight_budget>0);
      }
    }
  }
  return available_weight_budget;
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
EvalStatus BufferLevel::PreEvaluationCheck(
    const problem::PerDataSpace<std::size_t> working_set_sizes,
    const tiling::CompoundMask mask,
    const bool break_on_failure)
{
  (void) break_on_failure;

  bool success = true;
  std::ostringstream fail_reason;
  
  if (specs_.size.IsSpecified())
  {
    // Ugh. If we can do a distributed multicast from this level,
    // then the required size may be smaller. However, that depends
    // on the multicast factor etc. that we don't know at this point.
    // Use a very loose filter and fail this check only if there's
    // no chance that this mapping can fit.
    auto available_capacity = specs_.effective_size.Get();
    if (network_read_->DistributedMulticastSupported())
    {
      available_capacity *= specs_.instances.Get();
    }

    // Find the total capacity required by all un-masked data types.
    std::size_t required_capacity = 0;
    for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
    {
      if (mask[pvi])
      {
        required_capacity += working_set_sizes.at(problem::Shape::DataSpaceID(pvi));
      }
    }

    if (required_capacity > available_capacity)
    {
      success = false;
      fail_reason << "mapped tile size " << required_capacity << " exceeds buffer capacity "
                  << available_capacity;
    }
    else if (required_capacity < specs_.effective_size.Get()
             * specs_.min_utilization.Get())
    {
      success = false;
      fail_reason << "mapped tile size " << required_capacity << " is less than constrained "
                  << "minimum utilization " << specs_.effective_size.Get() * specs_.min_utilization.Get();
    }
  }

  EvalStatus eval_status;
  eval_status.success = success;
  eval_status.fail_reason = fail_reason.str();

  return eval_status;  
}

//
// Heavyweight Evaluate() function. Unused original timeloop function
// FIXME: Derive FanoutX, FanoutY, MeshX, MeshY from mapping if unspecified.
//
EvalStatus BufferLevel::Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                                 const std::uint64_t compute_cycles,
                                 const bool break_on_failure)
{
  auto eval_status = ComputeAccesses(tile, mask, break_on_failure);
  if (!break_on_failure || eval_status.success)
  {
    ComputeBufferEnergy();
    ComputeReductionEnergy();
    ComputeAddrGenEnergy();
    ComputePerformance(compute_cycles);
  }
  return eval_status;
}

bool BufferLevel::HardwareReductionSupported()
{
  // FIXME: take this information from an explicit arch spec.
  return !(specs_.technology.IsSpecified() &&
           specs_.technology.Get() == Technology::DRAM);
}

void BufferLevel::ConnectRead(std::shared_ptr<Network> network)
{
  network_read_ = network;
}

void BufferLevel::ConnectFill(std::shared_ptr<Network> network)
{
  network_fill_ = network;
}

void BufferLevel::ConnectUpdate(std::shared_ptr<Network> network)
{
  network_update_ = network;
}

void BufferLevel::ConnectDrain(std::shared_ptr<Network> network)
{
  network_drain_ = network;
}

//
// Heavyweight SparseEvaluate() function.
//
EvalStatus BufferLevel::SparseEvaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                           const std::uint64_t compute_cycles, const bool break_on_failure, double weight_sparsity, 
                           double unfilled, const Mapping& mapping, bool is_innermost_level, 
                           const problem::Workload& workload, double inverse_speedup,
                           unsigned long sorting_blk_ptr_access, const problem::DataPoint& subtile)
{
  is_innermost_level_ = is_innermost_level;
  auto eval_status = ComputeSparseAccesses(tile, mask, break_on_failure, weight_sparsity, /* overflow_breakdown,*/ unfilled, mapping, workload);
  if (!break_on_failure || eval_status.success)
  {
    ComputeIndexTransfer(workload, subtile);
    ComputeBufferEnergy();
    ComputeReductionEnergy();
    ComputeAddrGenEnergy();
    ComputePerformance(compute_cycles);
    ComputeSortingEnergy(sorting_blk_ptr_access);
    stats_.sparse_cycles = static_cast<std::uint64_t>(std::ceil(stats_.cycles * inverse_speedup));
  }

  return eval_status;
}

EvalStatus BufferLevel::ComputeAccesses(const tiling::CompoundTile& tile,
                                  const tiling::CompoundMask& mask,
                                  const bool break_on_failure,
                                  bool sparse) // we can turn this knob on/off to exploit conservative/aggressive buffer sizing!
{
  (void) break_on_failure;

  bool success = true;
  std::ostringstream fail_reason;
  
  // Subnest FSM should be same for each problem::Shape::DataSpaceID in the list,
  // so just copy it from datatype #0.
  subnest_ = tile[0].subnest;

  //
  // 1. Collect stats (stats are always collected per-DataSpaceID).
  //
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    stats_.keep[pv] = mask[pv];
    
    stats_.partition_size[pv] = tile[pvi].partition_size;
    stats_.utilized_capacity[pv] = tile[pvi].size;
    stats_.utilized_instances[pv] = tile[pvi].replication_factor;

    assert((tile[pvi].size == 0) == (tile[pvi].content_accesses == 0));

    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      // First epoch is an Update, all subsequent epochs are Read-Modify-Update.
      assert(tile[pvi].size == 0 || tile[pvi].content_accesses % tile[pvi].size == 0);

      stats_.reads[pv] = tile[pvi].content_accesses - tile[pvi].partition_size;
      stats_.updates[pv] = tile[pvi].content_accesses;
      stats_.fills[pv] = tile[pvi].fills;
      stats_.address_generations[pv] = stats_.updates[pv] + stats_.fills[pv]; // scalar

      // FIXME: temporal reduction and network costs if hardware reduction isn't
      // supported appears to be wonky - network costs may need to trickle down
      // all the way to the level that has the reduction hardware.
      stats_.temporal_reductions[pv] = tile[pvi].content_accesses - tile[pvi].partition_size;
    }
    else // Read-only data type.
    {
      stats_.reads[pv] = tile[pvi].content_accesses;
      stats_.updates[pv] = 0;
      stats_.fills[pv] = tile[pvi].fills;
      stats_.address_generations[pv] = stats_.reads[pv] + stats_.fills[pv]; // scalar
      stats_.temporal_reductions[pv] = 0;
    }
  }

  //
  // 2. Derive/validate architecture specs based on stats.
  //      
  auto total_utilized_capacity = std::accumulate(stats_.utilized_capacity.begin(),
                                                 stats_.utilized_capacity.end(),
                                                 0ULL);
  if (!specs_.size.IsSpecified())
  {
#ifdef UPDATE_UNSPECIFIED_SPECS
    specs_.size = std::ceil(total_utilized_capacity * specs_.multiple_buffering.Get());
#endif
  }
  // pass evaluation on whether dense working set fit if sparse true: aggressive 
  else if (!sparse && total_utilized_capacity > specs_.effective_size.Get())
  {
    success = false;
    fail_reason << "mapped tile size " << total_utilized_capacity << " exceeds buffer capacity "
                << specs_.effective_size.Get();
  }
  // pass evaluation on whether dense working set fit if sparse true: aggressive
  else if (!sparse && total_utilized_capacity < specs_.effective_size.Get()
           * specs_.min_utilization.Get())
  {
    success = false;
    fail_reason << "mapped tile size " << total_utilized_capacity << " is less than constrained "
                << "minimum utilization " << specs_.effective_size.Get() * specs_.min_utilization.Get();
  }

  assert (specs_.block_size.IsSpecified());
    
  assert (specs_.cluster_size.IsSpecified());
   
  // Compute address-generation bits.
  if (specs_.size.IsSpecified())
  {
    double address_range = std::ceil(static_cast<double>(specs_.size.Get() / specs_.block_size.Get()));
    specs_.addr_gen_bits = static_cast<unsigned long>(std::ceil(std::log2(address_range)));
  }
  else if (specs_.technology.Get() == Technology::SRAM)
  {
    // Use utilized capacity as proxy for size.
    double address_range = std::ceil(static_cast<double>(total_utilized_capacity / specs_.block_size.Get()));
    specs_.addr_gen_bits = static_cast<unsigned long>(std::ceil(std::log2(address_range)));
  }
  else // DRAM.
  {
#ifdef FIXED_DRAM_SIZE_IF_UNSPECIFIED
    // DRAM of un-specified size, use 48-bit physical address.
    specs_.addr_gen_bits = 48;
#else
    // Use utilized capacity as proxy for size.
    double address_range = std::ceil(static_cast<double>(total_utilized_capacity / specs_.block_size.Get()));
    specs_.addr_gen_bits = static_cast<unsigned long>(std::ceil(std::log2(address_range)));
#endif
  }

  if (!specs_.instances.IsSpecified())
  {
#ifdef UPDATE_UNSPECIFIED_SPECS
    specs_.instances = stats_.utilized_instances.Max();
#endif
  }
  else if (stats_.utilized_instances.Max() > specs_.instances.Get())
  {
    success = false;
    fail_reason << "mapped instances " << stats_.utilized_instances.Max() << " exceeds available hardware instances "
                << specs_.instances.Get();
  }

  // Bandwidth constraints cannot be checked/inherited at this point
  // because the calculation is a little more involved. We will do
  // this later in the ComputePerformance() function.      

  // Compute utilized clusters.
  // FIXME: should derive this from precise spatial mapping.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    // The following equation assumes fully condensed mapping. Do a ceil-div.
    // stats_.utilized_clusters[pv] = 1 + (stats_.utilized_instances[pv] - 1) /
    //    specs_.cluster_size.Get();
    // Assume utilized instances are sprinkled uniformly across all clusters.
    auto num_clusters = specs_.instances.Get() / specs_.cluster_size.Get();
    stats_.utilized_clusters[pv] = std::min(stats_.utilized_instances[pv],
                                            num_clusters);
  }

  is_evaluated_ = success;

  EvalStatus eval_status;
  eval_status.success = success;
  eval_status.fail_reason = fail_reason.str();
    
  return eval_status;
}

void BufferLevel::ComputeOverflow(unsigned storage_level, 
                                  std::vector<double> overflow_breakdown, 
                                  const tiling::CompoundMaskNest& bypass_nest)
{
  auto wid = problem::GetShape()->name == "Weight-Update-Depthwise" ? 
              (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : 
              (problem::GetShape()->DataSpaceNameToID.at("Weights"));
  // TODO: This is a bit ugly, we should change how overflow represented in stats
  // For now, convert multi-level access to equivalent next target level access
  auto next_non_bypass_level = specs_.next; 
  auto current_level = storage_level+1;
  while(!bypass_nest[wid].test(current_level))
  {
    next_non_bypass_level = next_non_bypass_level->next;
    current_level++;
  }
  assert(next_non_bypass_level != nullptr); // make sure we didn't go out of bounds
  auto next_level_cost = next_non_bypass_level->vector_access_energy.Get();
  specs_.next_non_bypass = next_non_bypass_level;

  converted_overflow_ = overflow_breakdown[0];
  overflow_access_ = overflow_breakdown[0];
  for(unsigned i=1;i<overflow_breakdown.size();i++)
  {
    next_non_bypass_level = next_non_bypass_level->next;
    current_level++;
    while(!bypass_nest[wid].test(current_level))
    {
      next_non_bypass_level = next_non_bypass_level->next;
      current_level++;
    }
    assert(next_non_bypass_level != nullptr);
    auto converted_cost = next_non_bypass_level->vector_access_energy.Get()/next_level_cost;
    converted_overflow_ += converted_cost * overflow_breakdown[i];
    overflow_access_ += overflow_breakdown[i];
  }
  // else // current non DRAM level already contains full WS

  assert(overflow_access_>=0);
  assert(converted_overflow_>=0);
}

// helper debug function
void BufferLevel::CheckNoOverflow()
{
  assert(overflow_access_ == 0.0);
  assert(converted_overflow_ == 0.0);
}

void BufferLevel::ResetOverflow()
{
  overflow_access_ = 0.0;
  converted_overflow_ = 0.0;
}

EvalStatus BufferLevel::ComputeSparseAccesses(const tiling::CompoundTile& tile,
                                  const tiling::CompoundMask& mask,
                                  const bool break_on_failure,
                                  double weight_sparsity,
                                  double unfilled, 
                                  const Mapping& mapping,
                                  const problem::Workload& workload) // for debug
{
  auto eval_status = ComputeAccesses(tile, mask, break_on_failure, true);

  // Check if Dense failed
  auto total_utilized_capacity = std::accumulate(stats_.utilized_capacity.begin(),
                                                 stats_.utilized_capacity.end(),
                                                 0ULL);
  if(specs_.effective_size.IsSpecified() && total_utilized_capacity > specs_.effective_size.Get())
    dense_success_ = false;

  // sanity check on sparse working set
  if(specs_.effective_size.IsSpecified() && weight_sparsity * total_utilized_capacity > specs_.effective_size.Get())
  {
    eval_status.success = false;
    eval_status.fail_reason = "average case overflows! Even the best load balancing strategy cannot help!";
    return eval_status;
  }

  // modify sparse tensor related specs
  auto wid = problem::GetShape()->name == "Weight-Update-Depthwise" ? 
              (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : 
              (problem::GetShape()->DataSpaceNameToID.at("Weights"));

  auto average_per_instance_converted_overflow = converted_overflow_/stats_.utilized_instances.at(wid);
  auto average_per_instance_overflow_access = overflow_access_/stats_.utilized_instances.at(wid);
  // auto average_per_instance_converted_overflow = converted_overflow_/stats_.utilized_instances.at(wid);
  // auto average_per_instance_overflow_access = overflow_access_/stats_.utilized_instances.at(wid);
  auto per_instance_unfill = unfilled / stats_.utilized_instances.at(wid);
  // maybe sanity check here with mapping and resulted access

  if(average_per_instance_overflow_access > stats_.reads[wid] * weight_sparsity)
  {
    std::cout <<std::endl;
    std::cout <<"Overflow should not exceed total access!"<< std::endl;

    std::cout << std::endl;
    std::cout <<"Per Instance Overflow: "<<average_per_instance_overflow_access<<std::endl;
    std::cout <<"Total Dense Reads: "<<stats_.reads[wid]<<std::endl;
    std::cout <<"Total Sparse Reads: "<<stats_.reads[wid] * weight_sparsity<<std::endl;
    std::cout <<"Troublesome Mapping: "<<std::endl;
    std::cout <<mapping<<std::endl;
    eval_status.success = false;
    eval_status.fail_reason = "Overflow should not exceed total access!";
    return eval_status;
  }

  if(per_instance_unfill > stats_.fills[wid] * weight_sparsity)
  {
    std::cout <<std::endl;
    std::cout <<"Unfills should not exceed total fills!"<< std::endl;
    std::cout <<"Per instance Unfills: "<<per_instance_unfill<<std::endl;
    std::cout <<"Total Dense Fills: "<<stats_.fills[wid]<<std::endl;
    std::cout <<"Total Sparse Fills: "<<stats_.fills[wid]* weight_sparsity<<std::endl;
    std::cout <<"Troublesome Mapping: "<<std::endl;
    std::cout <<mapping<<std::endl; 
    eval_status.success = false;
    eval_status.fail_reason = "Unfills should not exceed total fills!";
    return eval_status;
  }

  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    // TODO: the logic is a bit messy below, clean up at some point
    if(pvi==unsigned(wid)) // sparse tensor (i.e. weight for most cases)
    {
      if(gInfOverflowBuffer) // idealized case: no overflow from upper levels
      {
        stats_.overflow[pvi] = 0.0;
        stats_.converted_overflow[pvi] = 0.0;
        per_instance_unfill = 0.0;
      }
      else // overflow resolved accessing upper level
      {
        stats_.overflow[pvi] = average_per_instance_overflow_access;
        stats_.converted_overflow[pvi] = average_per_instance_converted_overflow;
      }
      
      stats_.sparse_reads[pvi] = stats_.reads[pvi] * weight_sparsity - stats_.overflow[pvi]; 
      
      stats_.sparse_updates[pvi] = stats_.updates[pvi];
      
      stats_.sparse_fills[pvi] = stats_.fills[pvi] * weight_sparsity - per_instance_unfill;
    } 
    else // non sparse tensor (i.e. activation for most cases)
    {
      if(problem::GetShape()->IsReadWriteDataSpace.at(pvi)) // read/write datatype
      {
        if(is_innermost_level_)
        {
          // determine read only activation id
          unsigned read_only_aid=0;
          for (unsigned i = 0; i < unsigned(problem::GetShape()->NumDataSpaces); i++)
            if(i != wid && i != pvi)
              read_only_aid = i;
          stats_.sparse_reads[pvi] = weight_sparsity * stats_.reads[pvi];
          stats_.sparse_updates[pvi] = weight_sparsity * stats_.updates[pvi];
          // we have fills when offloaded psum need to be refetched due to scheduling
          stats_.sparse_fills[pvi] = stats_.fills[pvi]; // what about this?
          stats_.overflow[pvi] = 0.0;
          stats_.converted_overflow[pvi] = 0.0;
        }
        else // not inner most level
        {
          stats_.sparse_reads[pvi] = stats_.reads[pvi];
          stats_.sparse_updates[pvi] = stats_.updates[pvi];
          stats_.sparse_fills[pvi] = stats_.fills[pvi];
          stats_.overflow[pvi] = 0.0;
          stats_.converted_overflow[pvi] = 0.0;
        }
      }
      else // input datatype: read only
      {
        stats_.sparse_reads[pvi] = stats_.reads[pvi];
        stats_.sparse_updates[pvi] = stats_.updates[pvi]; // or 0, no update on read only data
        stats_.sparse_fills[pvi] = stats_.fills[pvi];
        stats_.overflow[pvi] = 0.0;
        stats_.converted_overflow[pvi] = 0.0;
      }
    }

    // enable those assertions after debugging each specific buggy mapping
    assert(stats_.sparse_reads[pvi]>=0);
    assert(stats_.sparse_updates[pvi]>=0);
    assert(stats_.sparse_fills[pvi]>=0);
    assert(stats_.overflow[pvi]>=0);
  } 

  return eval_status;
}

// Compute buffer energy.
void BufferLevel::ComputeBufferEnergy()
{
  // NOTE! Stats are always maintained per-DataSpaceID
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    auto instance_accesses = stats_.reads.at(pv) + stats_.updates.at(pv) + stats_.fills.at(pv); // per instance access
    auto sparse_instance_accesses = stats_.sparse_reads.at(pv) + stats_.sparse_updates.at(pv) + stats_.sparse_fills.at(pv);
    auto sparse_instance_index_accesses = stats_.sparse_index_reads.at(pv) + stats_.sparse_index_updates.at(pv) + stats_.sparse_index_fills.at(pv);

    auto block_size = specs_.block_size.Get();
    double vector_accesses =
      (instance_accesses % block_size == 0) ?
      (instance_accesses / block_size)      :
      (instance_accesses / block_size) + 1;

    double sparse_vector_accesses = std::ceil(sparse_instance_accesses / block_size);
    double sparse_vector_index_accesses = std::ceil(sparse_instance_index_accesses / block_size);

    double cluster_access_energy = vector_accesses * 
      specs_.vector_access_energy.Get();

    double sparse_cluster_access_energy = sparse_vector_accesses * 
      specs_.vector_access_energy.Get();

    double sparse_cluster_index_access_energy = sparse_vector_index_accesses * 
      specs_.vector_access_energy.Get();
    
    // overflow should only for weights
    // add assersion on weights
    if(stats_.overflow[pv] != 0)
    {
      double vector_overflow_accesses = std::ceil(stats_.converted_overflow[pv] / block_size); 
      assert(vector_overflow_accesses>=0);

      stats_.overflow_energy[pv] = (specs_.next_non_bypass == nullptr) ? 0 :
        vector_overflow_accesses * specs_.next_non_bypass->vector_access_energy.Get();
    }
    else
      stats_.overflow_energy[pv] = 0;

    // Spread out the cost between the utilized instances in each cluster.
    // This is because all the later stat-processing is per-instance.
    if (stats_.utilized_instances.at(pv) > 0)
    {
      double cluster_utilization = double(stats_.utilized_instances.at(pv)) /
        double(stats_.utilized_clusters.at(pv));
      stats_.energy[pv] = cluster_access_energy / cluster_utilization;
      stats_.sparse_energy[pv] = (sparse_cluster_access_energy+stats_.overflow_energy[pv]) / cluster_utilization; // per instance? (what's cluster?)
      stats_.sparse_index_energy[pv] = sparse_cluster_index_access_energy / cluster_utilization;
      stats_.energy_per_access[pv] = stats_.energy.at(pv) / instance_accesses;
      stats_.sparse_energy_per_access[pv] = stats_.sparse_energy.at(pv) / sparse_instance_accesses;
    }
    else
    {
      stats_.energy[pv] = 0;
      stats_.sparse_energy[pv] = 0;
      stats_.sparse_index_energy[pv] = 0;
      stats_.energy_per_access[pv] = 0;
      stats_.sparse_energy_per_access[pv] = 0;
    }
    assert(stats_.sparse_energy[pv]>=0);
    assert(stats_.overflow_energy[pv]>=0);
    assert(stats_.sparse_index_energy[pv]>=0);
  } // for each dataspace
}

void BufferLevel::ComputeIndexTransfer(const problem::Workload& workload, const problem::DataPoint& subtile)
{
  /* Pointer Transfer */

  /* DRAM: block_ptr: need to know block size */
  // read: fw/bw for weight, wu for ifmap (emulated as weight as well), blocks are reordered to form subtile in GLB
  // write: fw for x, wu for dl/dw (emulated as output), created after quantile unit

  /* GLB: subtile_ptr: need to know subtile size */
  // read: fw/bw for weight, wu for ifmap (emulated as weight as well), subtile and it's pointer are passed to PE local RF
  // write: could only have psum for dl/dw (emulated as output), they are dense

  /* PE: We have linear access over subtile masks, and we execute subtile sequentially */

  auto ofid = problem::GetShape()->DataSpaceNameToID.at("Outputs");
  auto ifid = problem::GetShape()->DataSpaceNameToID.at("Inputs");

  auto wid = problem::GetShape()->name == "Weight-Update-Depthwise" ? 
                  (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : 
                  (problem::GetShape()->DataSpaceNameToID.at("Weights"));

  auto rid=problem::GetShape()->DimensionNameToID.at("R");
  auto sid=problem::GetShape()->DimensionNameToID.at("S");

  const unsigned blk_ptr_bits = 20; // 20 bits for naive case

  std::size_t read_block_size=0; // default: indicates no block pointer transfer
  std::size_t write_block_size=0;
  
  // Evaluate block size that determines # of bits in block mask
  bool is_wu = gEquivlentWU || problem::GetShape()->name == "Weight-Update-Depthwise";
  if(this->Name() == "DRAM")
  {
    if(is_wu) // in wu pass
    {
      // for ifmap (emulated as weight), block size == subtile size
      read_block_size = subtile.Volume();
      // for dl/dw
      write_block_size = workload.GetBound(rid) * workload.GetBound(sid);
      // write_mask = true;
    }
    else if(problem::GetShape()->IsReadWriteDataSpace.at(ofid)) // in fw
    {
      // for weight
      read_block_size = workload.GetBound(rid) * workload.GetBound(sid);
      // for x: we have to know WU subtile. write_block_size Currently ignored
    }
    else if(problem::GetShape()->IsReadWriteDataSpace.at(ifid)) // in bw
    {
      // for weight
      read_block_size = workload.GetBound(rid) * workload.GetBound(sid);
    }
  }
  else if(this->Name() == "GlobalBuffer")
  {
    // need to know subtile size in all passes
    read_block_size = subtile.Volume();
  }

  /* Mask Transfer */
  // accessing masks: cost is scaled from dense access, therfore it is robust to sparsity

  /* DRAM */
  // read: fw/bw for weight, wu for ifmap (emulated as weight as well), blocks masks are reordered to form subtile in GLB
  // write: fw for x, wu for dl/dw (emulated as output), created after quantile unit

  /* GLB: subtile_ptr: need to know subtile size */
  // read: fw/bw for weight, wu for ifmap (emulated as weight as well), subtile and it's pointer are passed to PE local RF
  // write: could only have psum for dl/dw (emulated as output), they are dense

  /* PE: We have linear access over subtile masks, and we execute subtile sequentially */
  // read: fw/bw for weight mask, wu for ifmap
  // write: we don't update mask

  /* RC encoding */
  // acceesing rc index: cost is scaled with sparse access, saving changes based on sparsity
  
  // index read, write, and update derived from dense equivalent and actual size
#define BLK_MASK 0 // maintain per block mask
#define RC_INDEX 1 // maintain row, column index for each nnz elements
#define ENCODING_SCHEME BLK_MASK

  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    if(pvi==unsigned(wid))
    {
#if ENCODING_SCHEME == BLK_MASK
      
      // Read
      auto mask_bits = read_block_size; // per block
      // figure out number of access first, then scale with bitwidth
      double bitwidth_scaling = static_cast<double>(mask_bits + blk_ptr_bits)/specs_.word_bits.Get();
      if(read_block_size > 0) // transfer block ptr and mask
      {
        stats_.sparse_index_reads[pvi] = stats_.reads[pvi] / read_block_size * bitwidth_scaling;
        stats_.sparse_index_fills[pvi] = stats_.fills[pvi] / read_block_size * bitwidth_scaling;
      }
      else // transfer mask only
      {
        stats_.sparse_index_reads[pvi] = stats_.reads[pvi] / specs_.word_bits.Get();
        stats_.sparse_index_fills[pvi] = stats_.fills[pvi] / specs_.word_bits.Get();
      }

      // Write
      mask_bits = write_block_size; // per block
      // figure out number of access first, then scale with bitwidth
      bitwidth_scaling = static_cast<double>(mask_bits + blk_ptr_bits)/specs_.word_bits.Get();
      
      if(write_block_size>0) // tranfer mask and ptr
      {
        stats_.sparse_index_updates[pvi] = stats_.updates[pvi] / write_block_size * bitwidth_scaling;
      }
// #elif ENCODING_SCHEME == RC_INDEX // currently not maintained !
//       auto rc_bits = std::ceil(std::log2(workload.GetBound(rid))) + std::ceil(std::log2(workload.GetBound(sid))); // per nnz element
//       double rc_scaling = static_cast<double>(rc_bits)/specs_.word_bits.Get();
//       double blk_ptr_scaling = static_cast<double>(blk_ptr_bits)/specs_.word_bits.Get();
//       stats_.sparse_index_reads[pvi] = stats_.sparse_reads[pvi] * rc_scaling + stats_.reads[pvi] / dense_block_size * blk_ptr_scaling;
//       stats_.sparse_index_updates[pvi] = stats_.sparse_updates[pvi] * rc_scaling + stats_.updates[pvi] / dense_block_size * blk_ptr_scaling;
//       stats_.sparse_index_fills[pvi] = stats_.sparse_fills[pvi] * rc_scaling + stats_.fills[pvi] / dense_block_size * blk_ptr_scaling;

#else
#error undefined ENCODING_SCHEME
#endif 

    }
    else
    {
      stats_.sparse_index_reads[pvi] = 0;
      stats_.sparse_index_updates[pvi] = 0;
      stats_.sparse_index_fills[pvi] = 0;
    }
    assert(stats_.sparse_index_reads[pvi]>=0);
    assert(stats_.sparse_index_updates[pvi]>=0);
    assert(stats_.sparse_index_fills[pvi]>=0);
  }
}

void BufferLevel::ComputeSortingEnergy(unsigned long sorting_blk_ptr_access)
{
  auto wid = problem::GetShape()->name == "Weight-Update-Depthwise" ? 
                  (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : 
                  (problem::GetShape()->DataSpaceNameToID.at("Weights"));
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    if(pvi==unsigned(wid))
    {

      auto ofid = problem::GetShape()->DataSpaceNameToID.at("Outputs");
      auto ifid = problem::GetShape()->DataSpaceNameToID.at("Inputs");

      auto block_size = specs_.block_size.Get();
      double sorting_blk_ptr_vector_accesses = 
        (sorting_blk_ptr_access % block_size == 0) ?
        (sorting_blk_ptr_access / block_size)      :
        (sorting_blk_ptr_access / block_size) + 1;
      
      bool is_wu = gEquivlentWU || problem::GetShape()->name == "Weight-Update-Depthwise";
      if(is_wu) // weight update
      {
        stats_.sorting_blk_ptr_energy[pvi] = specs_.vector_access_energy.Get() * sorting_blk_ptr_access;
      }
      if (problem::GetShape()->IsReadWriteDataSpace.at(ofid)) // forward pass
      {
        // assume we align with forward case
        stats_.sorting_blk_ptr_energy[pvi] = specs_.vector_access_energy.Get() * sorting_blk_ptr_access;
        // Every Access is a block access
      }
      else if (problem::GetShape()->IsReadWriteDataSpace.at(ifid)) // backward pass
      {
        // assume bw unaligned
        // we measure the worst case scenarios
        // consequtive blk ptrs have equil probability between lying in the same bank and different banks
        stats_.sorting_blk_ptr_energy[pvi] = 1.5 * specs_.vector_access_energy.Get() * sorting_blk_ptr_vector_accesses;
      }
    }
    else
      stats_.sorting_blk_ptr_energy[pvi] = 0;
  }
}

//
// Compute reduction energy.
//
void BufferLevel::ComputeReductionEnergy()
{
  // Temporal reduction: add a value coming in on the network to a value stored locally.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    if (problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      stats_.temporal_reduction_energy[pv] = stats_.temporal_reductions[pv] * 
        pat::AdderEnergy(specs_.word_bits.Get(), network_update_->WordBits());
    }
    else
    {
      stats_.temporal_reduction_energy[pv] = 0;
    }
  }
}

//
// Compute address generation energy.
//
void BufferLevel::ComputeAddrGenEnergy()
{
  // Note! Address-generation is amortized across the cluster width.
  // We compute the per-cluster energy here. When we sum across instances,
  // we need to be careful to only count each cluster once.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    // We'll use an addr-gen-bits + addr-gen-bits adder, though
    // it's probably cheaper than that. However, we can't assume
    // a 1-bit increment.
    auto pv = problem::Shape::DataSpaceID(pvi);
    stats_.addr_gen_energy[pv] = stats_.address_generations[pv] *
      pat::AdderEnergy(specs_.addr_gen_bits.Get(), specs_.addr_gen_bits.Get());
  }
}

//
// Compute performance.
//
void BufferLevel::ComputePerformance(const std::uint64_t compute_cycles)
{
  //
  // Step 1: Compute unconstrained bandwidth demand.
  //
  problem::PerDataSpace<double> unconstrained_read_bandwidth;
  problem::PerDataSpace<double> unconstrained_write_bandwidth;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    auto total_read_accesses    =   stats_.reads.at(pv);
    auto total_write_accesses   =   stats_.updates.at(pv) + stats_.fills.at(pv);
    unconstrained_read_bandwidth[pv]  = (double(total_read_accesses)  / compute_cycles);
    unconstrained_write_bandwidth[pv] = (double(total_write_accesses) / compute_cycles);
  }

  //  // Step 2: Compare vs. specified bandwidth and calculate slowdown.
  //
  stats_.slowdown = 1.0;

  // Find slowdown.
  auto total_unconstrained_read_bandwidth  = std::accumulate(unconstrained_read_bandwidth.begin(),  unconstrained_read_bandwidth.end(),  0.0);
  auto total_unconstrained_write_bandwidth = std::accumulate(unconstrained_write_bandwidth.begin(), unconstrained_write_bandwidth.end(), 0.0);

  if (specs_.read_bandwidth.IsSpecified() &&
      specs_.read_bandwidth.Get() < total_unconstrained_read_bandwidth)
  {
    stats_.slowdown =
      std::min(stats_.slowdown,
               specs_.read_bandwidth.Get() / total_unconstrained_read_bandwidth);
  }
  if (specs_.write_bandwidth.IsSpecified() &&
      specs_.write_bandwidth.Get() < total_unconstrained_write_bandwidth)
  {
    stats_.slowdown =
      std::min(stats_.slowdown,
               specs_.write_bandwidth.Get() / total_unconstrained_write_bandwidth);
  }

  //
  // Step 3:
  // Calculate real bandwidths based on worst slowdown. For shared buffers this
  // ends up effectively slowing down each datatype's bandwidth by the slowdown
  // amount, which is slightly weird but appears to be harmless.
  //
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    stats_.read_bandwidth[pv]  = stats_.slowdown * unconstrained_read_bandwidth.at(pv);
    stats_.write_bandwidth[pv] = stats_.slowdown * unconstrained_write_bandwidth.at(pv);
  }

  //
  // Step 4: Calculate execution cycles.
  //
  stats_.cycles = std::uint64_t(ceil(compute_cycles / stats_.slowdown));

  //
  // Step 5: Update arch specs.
  //
#ifdef UPDATE_UNSPECIFIED_SPECS
  if (!specs_.read_bandwidth.IsSpecified())
    specs_.read_bandwidth = std::accumulate(stats_.read_bandwidth.begin(), stats_.read_bandwidth.end(), 0.0);
  if (!specs_.write_bandwidth.IsSpecified())
    specs_.write_bandwidth = std::accumulate(stats_.write_bandwidth.begin(), stats_.write_bandwidth.end(), 0.0);
#endif
}

//
// Accessors.
//

STAT_ACCESSOR(double, BufferLevel, StorageEnergy, stats_.energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, BufferLevel, SparseStorageEnergy, stats_.sparse_energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, BufferLevel, SparseIndexEnergy, stats_.sparse_index_energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, BufferLevel, SortingEnergy, stats_.sorting_blk_ptr_energy.at(pv))
STAT_ACCESSOR(double, BufferLevel, TemporalReductionEnergy, stats_.temporal_reduction_energy.at(pv) * stats_.utilized_instances.at(pv))
STAT_ACCESSOR(double, BufferLevel, AddrGenEnergy, stats_.addr_gen_energy.at(pv) * stats_.utilized_clusters.at(pv)) // Note!!! clusters, not instances.
STAT_ACCESSOR(double, BufferLevel, Energy,
              StorageEnergy(pv) +
              TemporalReductionEnergy(pv) +
              AddrGenEnergy(pv))
STAT_ACCESSOR(double, BufferLevel, SparseEnergy,
              SparseStorageEnergy(pv) +
              SparseIndexEnergy(pv) +
              SortingEnergy(pv) +
              TemporalReductionEnergy(pv) +
              AddrGenEnergy(pv))

STAT_ACCESSOR(std::uint64_t, BufferLevel, Accesses, stats_.utilized_instances.at(pv) * (stats_.reads.at(pv) + stats_.updates.at(pv) + stats_.fills.at(pv)))
STAT_ACCESSOR(std::uint64_t, BufferLevel, UtilizedCapacity, stats_.utilized_capacity.at(pv))
STAT_ACCESSOR(std::uint64_t, BufferLevel, UtilizedInstances, stats_.utilized_instances.at(pv))

std::string BufferLevel::Name() const
{
  return specs_.name.Get();
}

double BufferLevel::Area() const
{
  double area = 0;
  area += specs_.storage_area.Get() * specs_.instances.Get();
  return area;
}

double BufferLevel::AreaPerInstance() const
{
  double area = 0;
  area += specs_.storage_area.Get();
  return area;
}

double BufferLevel::Size() const
{
  // FIXME: this is per-instance. This is inconsistent with the naming
  // convention of some of the other methods, which are summed across instances.
  double size = 0;
  size += specs_.size.Get();
  return size;
}

double BufferLevel::CapacityUtilization() const
{
  double utilized_capacity = 0;
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);
    utilized_capacity += stats_.utilized_capacity.at(pv) *
      stats_.utilized_instances.at(pv);
  }

  double total_capacity = Size() * specs_.instances.Get();

  return utilized_capacity / total_capacity;
}

std::uint64_t BufferLevel::Cycles() const
{
  return stats_.cycles;
}

std::uint64_t BufferLevel::SparseCycles() const
{
  return stats_.sparse_cycles;
}

// ---------------
//    Printers
// ---------------

std::ostream& operator<<(std::ostream& out, const BufferLevel::Technology& tech)
{
  switch (tech)
  {
    case BufferLevel::Technology::SRAM: out << "SRAM"; break;
    case BufferLevel::Technology::DRAM: out << "DRAM"; break;
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const BufferLevel& buffer_level)
{
  buffer_level.Print(out);
  return out;
}

void BufferLevel::Print(std::ostream& out) const
{
  std::string indent = "    ";

  auto& specs = specs_;
  auto& stats = stats_;

  // Print level name.
  out << "=== " << specs.level_name << " ===" << std::endl;  
  out << std::endl;

  // Print specs.
  out << indent << "SPECS" << std::endl;
  out << indent << "-----" << std::endl;

  out << indent << indent << "Technology           : " << specs.technology << std::endl;
  out << indent << indent << "Size                 : " << specs.size << std::endl;
  out << indent << indent << "Word bits            : " << specs.word_bits << std::endl;    
  out << indent << indent << "Block size           : " << specs.block_size << std::endl;
  out << indent << indent << "Cluster size         : " << specs.cluster_size << std::endl;
  out << indent << indent << "Instances            : " << specs.instances << " ("
      << specs.meshX << "*" << specs.meshY << ")" << std::endl;
  out << indent << indent << "Read bandwidth       : " << specs.read_bandwidth << std::endl;    
  out << indent << indent << "Write bandwidth      : " << specs.write_bandwidth << std::endl;    
  out << indent << indent << "Multiple buffering   : " << specs.multiple_buffering << std::endl;
  out << indent << indent << "Effective size       : " << specs.effective_size << std::endl;
  out << indent << indent << "Min utilization      : " << specs.min_utilization << std::endl;
  out << indent << indent << "Vector access energy : " << specs.vector_access_energy << " pJ" << std::endl;
  out << indent << indent << "Area                 : " << specs.storage_area << " um^2" << std::endl;

  out << std::endl;

  // If the buffer hasn't been evaluated on a specific mapping yet, return.
  if (!IsEvaluated())
  {
    return;
  }

  // Print mapping.
  out << indent << "MAPPING" << std::endl;
  out << indent << "-------" << std::endl;
  out << indent << "Loop nest:" << std::endl;
  std::string loopindent = "  ";
  for (auto loop = subnest_.rbegin(); loop != subnest_.rend(); loop++)
  {
    // Do not print loop if it's a trivial factor.
    if ((loop->start + loop->stride) < loop->end)
    {
      out << indent << loopindent << *loop << std::endl;
      loopindent += "  ";
    }
  }
  out << std::endl;

  // Print stats.
  out << indent << "STATS" << std::endl;
  out << indent << "-----" << std::endl;

  out << indent << "Cycles               : " << stats.cycles << std::endl;
  out << indent << "Sparse Cycles        : " << stats.sparse_cycles << std::endl;
  out << indent << "Bandwidth throttling : " << stats.slowdown << std::endl;

  const auto& dense_valid = dense_success_?"Valid":"Invalid";
  out << indent << "Dense Mapping Valid  : " << dense_valid << std::endl;
  
  // Print per-DataSpaceID stats.
  for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
  {
    auto pv = problem::Shape::DataSpaceID(pvi);

    if (stats.keep.at(pv))
    {
      out << indent << problem::GetShape()->DataSpaceIDToName.at(pv) << ":" << std::endl;

      out << indent + indent << "Partition size                           : " << stats.partition_size.at(pv) << std::endl;
      out << indent + indent << "Utilized capacity                        : " << stats.utilized_capacity.at(pv) << std::endl;
      out << indent + indent << "Utilized instances (max)                 : " << stats.utilized_instances.at(pv) << std::endl;
      out << indent + indent << "Utilized clusters (max)                  : " << stats.utilized_clusters.at(pv) << std::endl;
      out << indent + indent << "Dense Scalar reads (per-instance)        : " << stats.reads.at(pv) << std::endl;
      out << indent + indent << "Dense Scalar updates (per-instance)      : " << stats.updates.at(pv) << std::endl;
      out << indent + indent << "Dense Scalar fills (per-instance)        : " << stats.fills.at(pv) << std::endl;
      out << indent + indent << "Sparse Scalar reads (per-instance)       : " << stats.sparse_reads.at(pv) << std::endl;
      out << indent + indent << "Sparse Scalar updates (per-instance)     : " << stats.sparse_updates.at(pv) << std::endl;
      out << indent + indent << "Sparse Scalar fills (per-instance)       : " << stats.sparse_fills.at(pv) << std::endl;
      out << indent + indent << "Sparse Overflows accesses (per-instance) : " << stats.overflow.at(pv) << std::endl;
      out << indent + indent << "Temporal reductions (per-instance)       : " << stats.temporal_reductions.at(pv) << std::endl;
      out << indent + indent << "Address generations (per-cluster)        : " << stats.address_generations.at(pv) << std::endl;
      out << indent + indent << "SortingEnergy (total):                   : " << stats.sorting_blk_ptr_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Energy (per-scalar-access)               : " << stats.energy_per_access.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Sparse Energy (per-scalar-access)        : " << stats.sparse_energy_per_access.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Dense Energy (per-instance)              : " << stats.energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Sparse Energy (per-instance)             : " << stats.sparse_energy.at(pv)<< " pJ" << std::endl;
      out << indent + indent << "Overflow Energy (per-instance)           : " << stats.overflow_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Dense Energy (total)                     : " << stats.energy.at(pv) * stats.utilized_instances.at(pv)
          << " pJ" << std::endl;
      out << indent + indent << "Sparse Energy (total)                    : " << stats.sparse_energy.at(pv) * stats.utilized_instances.at(pv) // sparse energy include all levels
          << " pJ" << std::endl;
      out << indent + indent << "Temporal Reduction Energy (per-instance) : "
          << stats.temporal_reduction_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Temporal Reduction Energy (total)        : "
          << stats.temporal_reduction_energy.at(pv) * stats.utilized_instances.at(pv)
          << " pJ" << std::endl;
      out << indent + indent << "Address Generation Energy (per-cluster)  : "
          << stats.addr_gen_energy.at(pv) << " pJ" << std::endl;
      out << indent + indent << "Address Generation Energy (total)        : "
          << stats.addr_gen_energy.at(pv) * stats.utilized_clusters.at(pv)
          << " pJ" << std::endl;
      out << indent + indent << "Read Bandwidth (per-instance)            : " << stats.read_bandwidth.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Read Bandwidth (total)                   : " << stats.read_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (per-instance)           : " << stats.write_bandwidth.at(pv) << " words/cycle" << std::endl;
      out << indent + indent << "Write Bandwidth (total)                  : " << stats.write_bandwidth.at(pv) * stats.utilized_instances.at(pv) << " words/cycle" << std::endl;
    }
  }

  out << std::endl;
}

}  // namespace model
