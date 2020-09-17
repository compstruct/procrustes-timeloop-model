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

#include "mapping/nest.hpp"
#include "mapping/mapping.hpp"
#include "workload/per-problem-dimension.hpp"
#include "workload/data-masks.hpp"
#include "cnpy.h"

namespace analysis
{

class NestAnalysis
{
 private:
  // Cached copy of loop nest under evaluation (used for speedup).
  loop::Nest cached_nest;
  uint128_t mapping_id_; // debugging only
  tiling::CompoundMaskNest bypass_nest_;
  
  // Properties of the nest being analyzed (copied over during construction).
  std::vector<uint64_t> storage_tiling_boundaries_;

  // Live state.
  std::vector<analysis::LoopState> nest_state_; // derived from nest->loops
  std::vector<int> indices_;
  std::uint64_t num_epochs_;
  // Identifies the spatial element
  // whose working set is currently being computed.
  // Dynamically updated by recursive calls.
  std::uint64_t spatial_id_;
  
  tiling::CompoundTileNest working_sets_;
  tiling::BodyInfo body_info_;
  tiling::SparsityInfo sparsity_info_;

  // Memoization structures to accelerate IndexToOperationPoint()
  std::vector<problem::PerProblemDimension<std::uint64_t>>
      per_level_dim_scales_;  // level * dim, can be used to infer operation space size
  problem::OperationPoint cur_transform_;
  std::vector<problem::OperationPoint> mold_low_;
  std::vector<problem::OperationPoint> mold_high_;
  // additional memoization structures
  std::vector<problem::DataPoint> level_high_; // size number of target levels
  std::vector<problem::DataPoint> memo_spatial_high_; // size number of target levels
  std::vector<unsigned> weight_multicast_; // size number of storage levels

  // std::string layer_name_;
  bool is_weight_mask_loaded_ = false;
  bool verbose_; // log out debugging info for evaluator and disable them in timeloop

  bool* weight_mask_data_;  
  std::vector<bool*> weight_snapshots_;
  std::vector<size_t> shape_;

  // support multiple snapshots
  unsigned total_samples_;
  unsigned sample_id_;
  std::vector<double> sample_sparsity_;
  double average_sparsity_;
  double maximum_sparsity_;

  // support better cycle modeling
  std::vector<double> speedup_; // speedup for each sample

  // Sorting support for load balancing
  bool is_partition_ = false;

  // Where latency goes when PE synchronized / not synchronized
  std::vector<unsigned> extra_element_hist_;
  // support level bypass
  std::vector<unsigned> target_storage_mapping_;

  std::vector<std::size_t> weights_working_set_; 
  std::vector<unsigned long> storage_level_tile_access_; // wws_tile_access_ across the whole storate level. 

  // buffer sparsity
  // storage levels * samples * spatial elements // TODO: maybe storage level * spatial elements * samples???
  std::vector<std::vector<std::vector<unsigned long>>> max_utilized_capacity_;
  std::vector<std::vector<std::vector<unsigned long>>> cumulative_utilized_capacity_;
  std::vector<std::vector<std::vector<double>>> average_utilized_capacity_;

  // tile sparsity less than analysis threshold will be guaranteed no overflow
  std::vector<double> analysis_threshold_;
  std::vector<size_t> available_weight_budget_; // this is intended to replace the prior one

  // overflow
  // storage levels * samples * spatial elements // maintain per sample stats is easier to debug
  typedef std::vector<double> MultiLevelOverflow; // across different storage levels
  std::vector<std::vector<std::vector<MultiLevelOverflow>>> cumulative_multi_level_overflow_amount_unbalanced_;
  std::vector<std::vector<std::vector<unsigned long>>> cumulative_overflow_amount_unbalanced_;

  // overflow frequency: unused for now, enable them if we need those stats
  std::vector<std::vector<std::vector<unsigned long>>> cumulative_overflow_tiles_unbalanced_; // later try measure this on subtile level

  struct OverflowMask // an inherit 1D vector which indexed as 3D (depthwise) and 4D (conv and FC) mask
  {
    std::vector<double> data;      // number of spatial elements at this level.
    problem::DataPoint shape;

    OverflowMask() { Reset(); }

    void Reset()
    {
      data.clear();
      shape.Reset();
    }
  };

  std::vector<std::vector<OverflowMask>> overflow_mask_unbalanced_; // storage level * samples

  std::vector<std::vector<MultiLevelOverflow>> cumulative_multi_level_overflow_amount_balanced_;
  std::vector<std::vector<unsigned long>> cumulative_overflow_amount_balanced_;
  std::vector<std::vector<unsigned long>> cumulative_overflow_tiles_balanced_;
  std::vector<std::vector<OverflowMask>> overflow_mask_balanced_;

  // per-level properties.
  std::vector<uint64_t> num_spatial_elems_;
  std::vector<uint64_t> spatial_fanouts_;

  // used to accelerate to IndexToOperationPoint computation
  // relevant only for master spatial levels.
  std::vector<uint64_t> horizontal_sizes_;
  std::vector<uint64_t> vertical_sizes_;

  // records if a level corresponds to the starting
  // point of a new storage tile.
  std::vector<bool> storage_boundary_level_; // a mask version of storage_tiling_boundaries_
  
  // any level which is at the transition point from temporal to
  // spatial nests is a master spatial level.
  // there should be one such level between each set of
  // consecutive physical storage levels.
  std::vector<bool> master_spatial_level_;
  
  // true if the spatial elements at a given master spatial
  // level are connected by on-chip links.
  std::vector<bool> linked_spatial_level_;

  bool working_sets_computed_ = false;

  problem::Workload* workload_ = nullptr;

  // Internal helper methods.
  void ComputeWorkingSets();

  void InitializeNestProperties();
  void InitNumSpatialElems();
  void InitStorageBoundaries();
  void InitSpatialFanouts();
  void InitPerLevelDimScales();

  void InitializeLiveState();
  
  void AnalysisSparseWorkingSet();
  void InitAnalysis();

  unsigned long AnalysisSpatialSparseBuffer(unsigned storage_level, unsigned target_level, 
                                  const problem::DataPoint& base, 
                                  const problem::DataPoint& mold_high, 
                                  const problem::DataPoint& mold_spatial_high);
  
  // overflow
  void OverflowAnalysis(unsigned storage_level,
                        unsigned long& cumulative_access,
                        unsigned long& cumulative_tiles,
                        MultiLevelOverflow& multi_level_access,
                        unsigned long overflow_capacity,
                        problem::DataPoint base);
  void InitOverflowMask(OverflowMask& mask, problem::DataPoint p);

  double GetOverflowMask(OverflowMask& mask, problem::DataPoint p);

  void SetOverflowMask(OverflowMask& mask, double value, problem::DataPoint p);

  void DisplayOverflowMask(OverflowMask& mask);
  
  unsigned CountMask(unsigned low0, unsigned low1, unsigned low2, unsigned low3,
                    unsigned high0, unsigned high1, unsigned high2, unsigned high3);
  unsigned CountMask(unsigned low0, unsigned low1, unsigned low2, 
                    unsigned high0, unsigned high1, unsigned high2); // depthwise
  unsigned CountMaskGeneric(problem::DataPoint low, problem::DataPoint high);
  unsigned CountMask(problem::DataPoint low, problem::DataPoint high);

  bool GetWeightMask(unsigned, unsigned, unsigned, unsigned); // for conv weights
  bool GetWeightMask(unsigned, unsigned, unsigned); // for depthwise weights
  // bool GetWeightMask(unsigned, unsigned); // for FC weights, deprecated
  bool GetWeightMaskGeneric(const problem::DataPoint& p); 

  // helper function introduced because of depthwise layer
  problem::DataPoint GetAuxiliaryDataPoint(unsigned ind, const problem::DataPoint& tile);
  unsigned GetWeightOrder();
  void SortingCost();
  void CollectWorkingSets();

  // private helper
  bool IsDWWU()
  {
    return workload_->GetShape()->name == "Weight-Update-Depthwise";
  }

  problem::OperationPoint IndexToOperationPoint_(const std::vector<int>& indices) const;
  
  problem::OperationSpace ComputeDeltas(
    std::vector<analysis::LoopState>::reverse_iterator cur, bool skip_delta = false);

  void ComputeTemporalWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                 problem::OperationSpace& point_set,
                                 analysis::ElementState& cur_state);
  void ComputeSpatialWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                problem::OperationSpace& point_set);

  void FillSpatialDeltas(std::vector<analysis::LoopState>::reverse_iterator cur,
                         std::vector<problem::OperationSpace>& spatial_deltas,
                         std::vector<bool>& valid_delta,
                         std::uint64_t base_index,
                         int depth = 0);

  void ComputeAccurateMulticastedAccesses(
      std::vector<analysis::LoopState>::reverse_iterator cur,
      const std::vector<problem::OperationSpace>& spatial_deltas,
      std::vector<problem::PerDataSpace<bool>>&
      unaccounted_delta,
      problem::PerDataSpace<std::vector<std::uint64_t>>& accesses,
      problem::PerDataSpace<std::vector<std::uint64_t>>& scatter_factors,
      problem::PerDataSpace<std::vector<std::uint64_t>>& cumulative_hops
    );

  void ComputeApproxMulticastedAccesses(
      std::vector<analysis::LoopState>::reverse_iterator cur,
      const std::vector<problem::OperationSpace>& spatial_deltas);

  void ComputeNetworkLinkTransfers(
      std::vector<analysis::LoopState>::reverse_iterator cur,
      const std::vector<problem::OperationSpace>& cur_spatial_deltas,
      std::vector<problem::PerDataSpace<bool>>&
      unaccounted_delta,
      problem::PerDataSpace<std::uint64_t>& link_transfers);
  
 public:  
  // API
  NestAnalysis();

  void Init(problem::Workload* wc, const Mapping& mapping, /* std::string layer_name=std::string(),*/ bool verbose=false);
  void Reset();
 
  std::vector<problem::PerDataSpace<std::size_t>> GetWorkingSetSizes_LTW() const;

  problem::PerDataSpace<std::vector<tiling::TileInfo>> GetWorkingSets(const std::vector<std::size_t>& available_budget);
  tiling::BodyInfo GetBodyInfo();
  tiling::SparsityInfo GetSparsityInfo();

  // weight loading
  void PreProcessLoadedWeights(const problem::DataMasks* mask);
  bool IsWeightMaskLoaded() {return is_weight_mask_loaded_;}
  double GetAverageSparsity() {return average_sparsity_;}
  double GetMaximumSparsity() {return maximum_sparsity_;}

  // Serialization.
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0) 
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(nest_state_);
      ar& boost::serialization::make_nvp("work_sets_",boost::serialization::make_array(working_sets_.data(),working_sets_.size()));
      ar& BOOST_SERIALIZATION_NVP(working_sets_computed_);
      // ar& BOOST_SERIALIZATION_NVP(compute_cycles_);
    }
  }

  friend std::ostream& operator << (std::ostream& out, const NestAnalysis& n);  
};

} // namespace analysis
