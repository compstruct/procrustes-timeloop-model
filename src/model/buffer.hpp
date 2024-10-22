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

#pragma once

#include <iostream>
#include <boost/serialization/export.hpp>

#include "model/model-base.hpp"
#include "model/level.hpp"
#include "loop-analysis/tiling.hpp"
#include "mapping/nest.hpp"
#include "mapping/mapping.hpp"
#include "compound-config/compound-config.hpp"
#include "model/util.hpp"
#include "model/network.hpp"

namespace model
{

//--------------------------------------------//
//                 BufferLevel                //
//--------------------------------------------//

class BufferLevel : public Level
{

  //
  // Types.
  //

 public:
  
  // Memory technology (FIXME: separate latch arrays).
  enum class Technology { SRAM, DRAM };
  friend std::ostream& operator<<(std::ostream& out, const Technology& tech);

  //
  // Specs.
  //
  struct Specs : public LevelSpecs
  {
    static const std::uint64_t kDefaultWordBits = 16;
    const std::string Type() const override { return "BufferLevel"; }
    
    // connect buffer levels for overflow
    std::shared_ptr<BufferLevel::Specs> next;
    // connect to next non bypass weight level for weight overflow
    std::shared_ptr<BufferLevel::Specs> next_non_bypass;

    Attribute<std::string> name;
    Attribute<Technology> technology;
    Attribute<std::uint64_t> size;
    Attribute<std::uint64_t> word_bits;
    Attribute<std::uint64_t> addr_gen_bits;
    Attribute<std::uint64_t> block_size;
    Attribute<std::uint64_t> cluster_size;
    Attribute<std::uint64_t> instances;    
    Attribute<std::uint64_t> meshX;
    Attribute<std::uint64_t> meshY;
    Attribute<double> read_bandwidth;
    Attribute<double> write_bandwidth;
    Attribute<double> multiple_buffering;
    Attribute<std::uint64_t> effective_size;
    Attribute<double> min_utilization;
    Attribute<std::uint64_t> num_ports;
    Attribute<std::uint64_t> num_banks;

    Attribute<std::string> read_network_name;
    Attribute<std::string> fill_network_name;
    Attribute<std::string> drain_network_name;
    Attribute<std::string> update_network_name;    

    // Physical Attributes (derived from technology model).
    // FIXME: move into separate struct?
    Attribute<double> vector_access_energy; // pJ
    Attribute<double> storage_area; // um^2

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(LevelSpecs);
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(name);
        ar& BOOST_SERIALIZATION_NVP(technology);
        ar& BOOST_SERIALIZATION_NVP(size);
        ar& BOOST_SERIALIZATION_NVP(word_bits);
        ar& BOOST_SERIALIZATION_NVP(addr_gen_bits);
        ar& BOOST_SERIALIZATION_NVP(block_size);
        ar& BOOST_SERIALIZATION_NVP(cluster_size);
        ar& BOOST_SERIALIZATION_NVP(instances);    
        ar& BOOST_SERIALIZATION_NVP(meshX);
        ar& BOOST_SERIALIZATION_NVP(meshY);
        ar& BOOST_SERIALIZATION_NVP(read_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(write_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(multiple_buffering);
        ar& BOOST_SERIALIZATION_NVP(min_utilization);
        ar& BOOST_SERIALIZATION_NVP(num_ports);
        ar& BOOST_SERIALIZATION_NVP(num_banks);

        ar& BOOST_SERIALIZATION_NVP(read_network_name);
        ar& BOOST_SERIALIZATION_NVP(fill_network_name);
        ar& BOOST_SERIALIZATION_NVP(drain_network_name);
        ar& BOOST_SERIALIZATION_NVP(update_network_name);
      }
    }

   public:
    std::shared_ptr<LevelSpecs> Clone() const override
    {
      return std::static_pointer_cast<LevelSpecs>(std::make_shared<Specs>(*this));
    }

  };
  
  //
  // Stats.
  //
  struct Stats
  {
    problem::PerDataSpace<bool> keep;
    problem::PerDataSpace<std::uint64_t> partition_size;
    problem::PerDataSpace<std::uint64_t> utilized_capacity;
    problem::PerDataSpace<std::uint64_t> utilized_instances;
    problem::PerDataSpace<std::uint64_t> utilized_clusters;
    problem::PerDataSpace<unsigned long> reads;
    problem::PerDataSpace<double> sparse_reads;
    problem::PerDataSpace<double> sparse_index_reads;
    problem::PerDataSpace<double> overflow; // total overflow count across levels (not so useful)
    problem::PerDataSpace<double> converted_overflow; // total overflow count converted to equivalent current level overflow count
    problem::PerDataSpace<unsigned long> updates;
    problem::PerDataSpace<double> sparse_updates;
    problem::PerDataSpace<double> sparse_index_updates;
    problem::PerDataSpace<unsigned long> fills;
    problem::PerDataSpace<double> sparse_fills;
    problem::PerDataSpace<double> sparse_index_fills;
    problem::PerDataSpace<unsigned long> address_generations;
    problem::PerDataSpace<unsigned long> temporal_reductions;
    problem::PerDataSpace<double> read_bandwidth;
    problem::PerDataSpace<double> write_bandwidth;
    problem::PerDataSpace<double> energy_per_access;
    problem::PerDataSpace<double> sparse_energy_per_access; 
    problem::PerDataSpace<double> energy;
    problem::PerDataSpace<double> overflow_energy;
    problem::PerDataSpace<double> sorting_blk_ptr_energy;
    problem::PerDataSpace<double> sparse_energy; // per instance
    problem::PerDataSpace<double> sparse_index_energy; // per instance
    problem::PerDataSpace<double> temporal_reduction_energy;
    problem::PerDataSpace<double> addr_gen_energy;

    std::uint64_t cycles;
    std::uint64_t sparse_cycles;
    double slowdown;

    // Serialization
    friend class boost::serialization::access;

    template <class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      if (version == 0)
      {
        ar& BOOST_SERIALIZATION_NVP(keep);
        ar& BOOST_SERIALIZATION_NVP(partition_size);
        ar& BOOST_SERIALIZATION_NVP(utilized_capacity);
        ar& BOOST_SERIALIZATION_NVP(utilized_instances);
        ar& BOOST_SERIALIZATION_NVP(utilized_clusters);
        ar& BOOST_SERIALIZATION_NVP(reads);
        ar& BOOST_SERIALIZATION_NVP(sparse_reads);
        ar& BOOST_SERIALIZATION_NVP(sparse_index_reads);
        ar& BOOST_SERIALIZATION_NVP(overflow);
        ar& BOOST_SERIALIZATION_NVP(converted_overflow);
        ar& BOOST_SERIALIZATION_NVP(updates);
        ar& BOOST_SERIALIZATION_NVP(sparse_updates);
        ar& BOOST_SERIALIZATION_NVP(sparse_index_updates);
        ar& BOOST_SERIALIZATION_NVP(fills);
        ar& BOOST_SERIALIZATION_NVP(sparse_fills);
        ar& BOOST_SERIALIZATION_NVP(sparse_index_fills);
        ar& BOOST_SERIALIZATION_NVP(address_generations);
        ar& BOOST_SERIALIZATION_NVP(temporal_reductions);
        ar& BOOST_SERIALIZATION_NVP(read_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(write_bandwidth);
        ar& BOOST_SERIALIZATION_NVP(energy_per_access);
        ar& BOOST_SERIALIZATION_NVP(sparse_energy_per_access);
        ar& BOOST_SERIALIZATION_NVP(energy);
        ar& BOOST_SERIALIZATION_NVP(overflow_energy);
        ar& BOOST_SERIALIZATION_NVP(sorting_blk_ptr_energy);
        ar& BOOST_SERIALIZATION_NVP(sparse_energy);
        ar& BOOST_SERIALIZATION_NVP(sparse_index_energy);
        ar& BOOST_SERIALIZATION_NVP(temporal_reduction_energy);
        ar& BOOST_SERIALIZATION_NVP(addr_gen_energy);
        ar& BOOST_SERIALIZATION_NVP(cycles);
        ar& BOOST_SERIALIZATION_NVP(sparse_cycles);
        ar& BOOST_SERIALIZATION_NVP(slowdown);
      }
    }
  };

  //
  // Data
  //
  
 private:

  std::vector<loop::Descriptor> subnest_;
  Stats stats_;
  Specs specs_;

  // put below into some struct like above?
  bool is_innermost_level_;
  double overflow_access_; // isolate overflow evaluation
  double converted_overflow_; // used for overflow energy across levels
  bool dense_success_=true;

  // Network endpoints.
  std::shared_ptr<Network> network_read_;
  std::shared_ptr<Network> network_fill_;
  std::shared_ptr<Network> network_update_;
  std::shared_ptr<Network> network_drain_;

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(Level);
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(subnest_);
      ar& BOOST_SERIALIZATION_NVP(specs_);
      ar& BOOST_SERIALIZATION_NVP(stats_);
    }
  }

  //
  // Private helpers.
  //

 private:
  EvalStatus ComputeAccesses(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                             const bool break_on_failure, bool sparse=false);
  EvalStatus ComputeSparseAccesses(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                       const bool break_on_failure, double weight_sparsity, double unfilled, 
                       const Mapping& mapping, const problem::Workload& workload);
  void ComputeIndexTransfer(const problem::Workload& workload, const problem::DataPoint& subtile);
  void ComputeSortingEnergy(unsigned long sorting_blk_ptr_access);

  void ComputePerformance(const std::uint64_t compute_cycles);
  void ComputeBufferEnergy();
  void ComputeReductionEnergy();
  void ComputeAddrGenEnergy();

  double StorageEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;

  double SparseStorageEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double SparseIndexEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  // We can change blocks level data layout during data filling from DRAM to on-chip buffer
  // Thus get rid of accessing every block ptr for reg file working set size on PEs
  double SortingEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double NetworkEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;

  double TemporalReductionEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;
  double AddrGenEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const;

  //
  // API
  //

 public:
  BufferLevel();
  BufferLevel(const Specs & specs);
  ~BufferLevel();

  std::shared_ptr<Level> Clone() const override
  {
    return std::static_pointer_cast<Level>(std::make_shared<BufferLevel>(*this));
  }

  // The hierarchical ParseSpecs functions are static and do not
  // affect the internal specs_ data structure, which is set by
  // the constructor when an object is actually created.
  static Specs ParseSpecs(config::CompoundConfigNode setting, uint32_t n_elements);
  static void ParseBufferSpecs(config::CompoundConfigNode buffer, uint32_t n_elements,
                               problem::Shape::DataSpaceID pv, Specs& specs);
  static void ValidateTopology(BufferLevel::Specs& specs);

  Specs& GetSpecs() { return specs_; }
  
  std::size_t AvailableBudget(const problem::PerDataSpace<std::size_t> working_set_sizes, const tiling::CompoundMask mask);

  bool HardwareReductionSupported() override;

  // Connect to networks.
  void ConnectRead(std::shared_ptr<Network> network);
  void ConnectFill(std::shared_ptr<Network> network);
  void ConnectUpdate(std::shared_ptr<Network> network);
  void ConnectDrain(std::shared_ptr<Network> network);
  std::shared_ptr<Network> GetReadNetwork() { return network_read_; }
  
  // Evaluation functions.
  EvalStatus PreEvaluationCheck(const problem::PerDataSpace<std::size_t> working_set_sizes,
                          const tiling::CompoundMask mask,
                          const bool break_on_failure) override;
  EvalStatus Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                      const std::uint64_t compute_cycles,
                      const bool break_on_failure) override;

  EvalStatus SparseEvaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                const std::uint64_t compute_cycles,
                const bool break_on_failure, double weight_sparsity, double unfilled,
                const Mapping& mapping, bool is_innermost_level, 
                const problem::Workload& workload, double inverse_speedup,
                unsigned long sorting_blk_ptr_access, const problem::DataPoint& subtile);

  // Evaluate Overflow
  void ComputeOverflow(unsigned storage_level, 
                      std::vector<double> overflow_breakdown, 
                      const tiling::CompoundMaskNest& bypass_nest);
  void CheckNoOverflow();
  void ResetOverflow();

  // Accessors (post-evaluation).
  
  double Energy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  double SparseEnergy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;

  bool GetDenseSuccess() const {return dense_success_;}
 
  std::string Name() const override;
  double Area() const override;
  double AreaPerInstance() const override;
  double Size() const;
  std::uint64_t Cycles() const override;
  std::uint64_t SparseCycles() const override;
  std::uint64_t Accesses(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  double CapacityUtilization() const override;
  std::uint64_t UtilizedCapacity(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  std::uint64_t UtilizedInstances(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const override;
  
  // Printers.
  void Print(std::ostream& out) const;
  friend std::ostream& operator << (std::ostream& out, const BufferLevel& buffer_level);
};

}  // namespace model
