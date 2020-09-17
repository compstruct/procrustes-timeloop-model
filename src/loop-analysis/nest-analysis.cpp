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

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <random>
#include <numeric>

// FIXME: num_spatial_elems, spatial_fanouts, replication_factor etc. are
//        all maintained across datatypes. They should be per-datatype at
//        this analytical level of abstraction. It's only when we get to
//        the architecture level that hardware may map multiple datatypes
//        on to the same storage and network structures.

// FIXME: Spatial model is X/Y only. Fortunately, generalizing this isn't
//        too difficult (as far as this module is concerned) since it's
//        limited to the ComputeNetworkLinkTransfers() function.

#include "util/misc.hpp"

#include "nest-analysis.hpp"

extern bool gTerminateEval;

bool gEnableLinkTransfers = (getenv("TIMELOOP_ENABLE_LINK_TRANSFERS") != NULL);
bool gEnableLinkTransferWarning = false;
bool gExtrapolateUniformTemporal = true;
bool gExtrapolateUniformSpatial = (getenv("TIMELOOP_DISABLE_SPATIAL_EXTRAPOLATION") == NULL);
bool gSyncPE = (getenv("TIMELOOP_USE_SYNC_PE") != NULL);
bool gGlobalSubtileSorting = (getenv("TIMELOOP_GLOBAL_SORT") != NULL);

namespace analysis
{

NestAnalysis::NestAnalysis()
{
}

void NestAnalysis::Init(problem::Workload* wc, const Mapping& mapping, bool verbose)
{
  auto nest = &mapping.loop_nest;
  mapping_id_ = mapping.id;
  bypass_nest_ = mapping.datatype_bypass_nest; 

  assert(nest != NULL);
  assert(wc != NULL);

  workload_ = wc;
  verbose_ = verbose;

  if (working_sets_computed_ && cached_nest == *nest)
  {
    // We've already worked on an identical nest before.
  }
  else
  {
    Reset();
    cached_nest = *nest;

    // Copy over everything we need from the nest.
    storage_tiling_boundaries_ = nest->storage_tiling_boundaries;

    if(verbose_)
    {
      // debug print
      std::cout<<"mapping: "<<std::endl;
      std::cout <<mapping<<std::endl;
      // storage
      std::cout<<"Storage tiling boundary looks like this: "<<std::endl;
      for (const auto &element: storage_tiling_boundaries_){
        std::cout<<element<<" ";
      }
      std::cout<<std::endl;
    }

    // Construct nest_state_.
    for (auto descriptor: nest->loops)
    {
      analysis::LoopState cur;
      if (nest_state_.size() == 0)
      {
        cur.level = 0; // loop level
      }
      else
      {
        cur.level = nest_state_.back().level + 1;
      }
      cur.descriptor = descriptor;
      nest_state_.push_back(cur);    
    }
  }
}

//
// Reset(): torpedo everything.
//
void NestAnalysis::Reset()
{
  storage_tiling_boundaries_.clear();
  
  nest_state_.clear();
  indices_.clear();
  num_epochs_ = 0;

  spatial_id_ = 0;
  for (auto& tile_nest: working_sets_)
  {
    tile_nest.clear();
  }

  per_level_dim_scales_.clear();
  cur_transform_ = problem::OperationPoint();
  mold_low_.clear();
  mold_high_.clear();

  // additional memoization
  level_high_.clear();
  memo_spatial_high_.clear();
  weight_multicast_.clear();

  // sample info
  sample_id_=0;

  // tile access count
  weights_working_set_.clear();
  storage_level_tile_access_.clear();

  // speedup
  speedup_.clear();

  extra_element_hist_.clear();

  // buffer sparsity
  max_utilized_capacity_.clear();
  cumulative_utilized_capacity_.clear();
  average_utilized_capacity_.clear();

  // overflow 
  cumulative_multi_level_overflow_amount_unbalanced_.clear();
  cumulative_overflow_amount_unbalanced_.clear();
  cumulative_overflow_tiles_unbalanced_.clear();
  // overflow_freq_.clear();
  overflow_mask_unbalanced_.clear();

  // overflow load balacing
  cumulative_multi_level_overflow_amount_balanced_.clear();
  cumulative_overflow_amount_balanced_.clear();
  cumulative_overflow_tiles_balanced_.clear();
  overflow_mask_balanced_.clear();

  num_spatial_elems_.clear();
  spatial_fanouts_.clear();

  horizontal_sizes_.clear();
  vertical_sizes_.clear();

  storage_boundary_level_.clear();
  master_spatial_level_.clear();
  linked_spatial_level_.clear();

  working_sets_computed_ = false;
  
  body_info_.Reset();
  sparsity_info_.Reset();
}

// Ugly function for pre-checking capacity fits before running the heavyweight
// ComputeWorkingSets() algorithm. FIXME: Integrate with ComputeWorkingSets().
std::vector<problem::PerDataSpace<std::size_t>>
NestAnalysis::GetWorkingSetSizes_LTW() const // in PreEvaluationCheck
{
  std::vector<problem::PerDataSpace<std::size_t>> working_set_sizes;

  problem::OperationPoint origin;
  problem::OperationPoint dimension_sizes;
  dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

  unsigned tiling_level = 0;
  for (unsigned loop_level = 0; loop_level < nest_state_.size(); loop_level++)
  {
    // nest_state_: derived from nest->loops
    auto & loop = nest_state_.at(loop_level).descriptor;
    ASSERT(loop.stride == 1);
    dimension_sizes[loop.dimension] *= loop.end;
        
    if (loop_level == storage_tiling_boundaries_.at(tiling_level))
    {
      // origin gives us the low corner (inclusive) of the operation space.
      // dimension_sizes gives the high corner (exclusive) of the operation space.
      // We need the inclusive high corner to build the operation space. See
      // OperationSpace constructor for details.
      problem::OperationPoint high = dimension_sizes;
      high.IncrementAllDimensions(-1); 
      // Note: high corner for DataSpace (AAHR) is exclusive, while inclusive for OperationSpace for accurate projection
      problem::OperationSpace maxtile(workload_, origin, high);
      working_set_sizes.push_back(maxtile.GetSizes());
      tiling_level++;
    }
  }

  ASSERT(working_set_sizes.size() == storage_tiling_boundaries_.size());
  return working_set_sizes;
}

problem::PerDataSpace<std::vector<tiling::TileInfo>>
NestAnalysis::GetWorkingSets(const std::vector<std::size_t>& available_budget)
{
  if (!working_sets_computed_)
  {
    available_weight_budget_.clear();
    for(unsigned int i = 0; i < available_budget.size()-1; i++) // exclude DRAM where always contains inf budget
      if(available_budget[i] > 0)
        available_weight_budget_.push_back(available_budget[i]);
    ComputeWorkingSets();
  }
  ASSERT(working_sets_computed_);
  return working_sets_;
}

tiling::BodyInfo NestAnalysis::GetBodyInfo()
{
  if (!working_sets_computed_)
  {
    ComputeWorkingSets();
  }
  ASSERT(working_sets_computed_);
  return body_info_;
}

tiling::SparsityInfo NestAnalysis::GetSparsityInfo()
{
  if (!working_sets_computed_)
  {
    ComputeWorkingSets();
  }
  ASSERT(working_sets_computed_);
  return sparsity_info_;
}

std::ostream& operator << (std::ostream& out, const NestAnalysis& n)
{
  for (auto cur = n.nest_state_.rbegin(); cur != n.nest_state_.rend(); cur++)
  {
    cur->descriptor.Print(out, false);
  }
  out << std::endl;
  return out;
}

void NestAnalysis::ComputeWorkingSets()
{
  // nest_state_ same length as nest->loops with live info
  if (nest_state_.size() != 0)
  {
    InitializeNestProperties();
    InitializeLiveState();
    AnalysisSparseWorkingSet(); // No ordering, otherwise need something like ComputeWorkingSetsRecursive_()

    // Recursive call starting from the last element of the list.
    num_epochs_ = 1;
    ComputeDeltas(nest_state_.rbegin());// return reverse iterator to reverse beginning
    // Analysed from the most outer loop to inner loop (from parent tile to child tile)

    CollectWorkingSets();
  }

  // Done.
  working_sets_computed_ = true;
}

// Internal helper methods

void NestAnalysis::InitializeNestProperties()
{
  InitNumSpatialElems();
  InitStorageBoundaries();
  InitSpatialFanouts();
  InitPerLevelDimScales();
}

void NestAnalysis::InitializeLiveState()
{
  indices_.resize(nest_state_.size());
  spatial_id_ = 0;
  
  body_info_.Reset();

  // from outer to inner
  for (auto loop = nest_state_.rbegin(); loop != nest_state_.rend(); loop++)
  {
    if (!loop::IsSpatial(loop->descriptor.spacetime_dimension) ||
        master_spatial_level_[loop->level]) // condition for all except non-master spatial levels
    {
      // we don't need live state for non-master spatial levels
      loop->live_state.resize(num_spatial_elems_[loop->level]);
    }

    for (auto& it : loop->live_state) // it is class ElementState
    {
      it.Reset();
      for (auto& acc : it.accesses)  // for each data space (problem variable)
      {
        acc.resize(spatial_fanouts_[loop->level]);
      }
      for (auto& sf : it.scatter_factors)
      {
        sf.resize(spatial_fanouts_[loop->level]);
      }
      for (auto& ch : it.cumulative_hops)
      {
        ch.resize(spatial_fanouts_[loop->level]);
      }

      if (linked_spatial_level_[loop->level]) // master spatial level
      {
        it.prev_point_sets.resize(analysis::ElementState::MAX_TIME_LAPSE);
        for (auto& elem : it.prev_point_sets)
        {
          elem.resize(spatial_fanouts_[loop->level]);
        }
      }
    }
  }
}

// might get rid of this also
// now keep it for compatibility
void NestAnalysis::PreProcessLoadedWeights(const problem::DataMasks* mask)
{
  shape_ = mask->getDataMaskShape();
  weight_mask_data_ = mask->getMaskPtr();
  total_samples_ = mask->getTotalSamples();

  if(verbose_)
  {
    std::cout <<"Weight mask loaded with the following size:"<<std::endl;
    for(auto dim : shape_)
      std::cout << dim << " ";
    std::cout << std::endl;
  }

  unsigned size = std::accumulate(shape_.begin(), shape_.end(), static_cast<unsigned>(1), std::multiplies<>());
  
  weight_snapshots_.resize(total_samples_);
  sample_sparsity_.resize(total_samples_);
  for(unsigned i = 0; i<total_samples_;i++)
  {
    weight_snapshots_[i] = weight_mask_data_ + i*size*sizeof(bool);

    unsigned num_mask=0;
    for(unsigned j=0; j<size;j++)
      if(weight_snapshots_[i][j]) // note that sparsity is not affected by transpose or not
        num_mask++;
    
    sample_sparsity_[i]=static_cast<double>(num_mask)/size;
    if(verbose_)
      std::cout <<"Total Mask Sparsity for sample "<<i<<" : "<<sample_sparsity_[i]<<std::endl;
  }

  average_sparsity_ = std::accumulate(sample_sparsity_.begin(), sample_sparsity_.end(), 0.0)/total_samples_;
  maximum_sparsity_ = *std::max_element(sample_sparsity_.begin(), sample_sparsity_.end());
  if(verbose_)
  {
    std::cout << std::endl;
    std::cout <<"average sample sparsity: "<<average_sparsity_<<std::endl;
    std::cout <<"maximum sample sparsity: "<<maximum_sparsity_<<std::endl;
  }

  // set weight mask load flag
  is_weight_mask_loaded_ = true;
}

// count # of sparse mask in a tile. Naive but efficient
/* inclusive low and exclusive high */
unsigned NestAnalysis::CountMask(unsigned low0, unsigned low1, unsigned low2, unsigned low3,
                                 unsigned high0, unsigned high1, unsigned high2, unsigned high3)
{
  unsigned counts=0;
  for(auto dim0 = low0; dim0 < high0; dim0++)
    for(auto dim1 = low1; dim1 < high1; dim1++)
      for(auto dim2 = low2; dim2 < high2; dim2++)
        for(auto dim3 = low3; dim3 < high3; dim3++)
          if(GetWeightMask(dim0, dim1, dim2, dim3))
            counts++;
  return counts;
}

// count # of sparse mask in a tile. Naive but efficient
/* inclusive low and exclusive high */
unsigned NestAnalysis::CountMask(unsigned low0, unsigned low1, unsigned low2, 
                                 unsigned high0, unsigned high1, unsigned high2)
{
  unsigned counts=0;
  for(auto dim0 = low0; dim0 < high0; dim0++)
    for(auto dim1 = low1; dim1 < high1; dim1++)
      for(auto dim2 = low2; dim2 < high2; dim2++)
        if(GetWeightMask(dim0, dim1, dim2))
            counts++;
  return counts;
}

/* inclusive low and exclusive high */
unsigned NestAnalysis::CountMask(problem::DataPoint low, problem::DataPoint high)
{
  unsigned counts=0;

  assert(low.Order() == high.Order());
  assert(low[0]<high[0] && low[1]<high[1]&&low[2]<high[2]);
  
  if(low.Order() == 3)
    counts=CountMask(low[0], low[1], low[2],
                     high[0],high[1],high[2]);
  else if(low.Order() == 4)
  {
    assert(low[3]<high[3]);
    counts=CountMask(low[0], low[1], low[2], low[3],
                     high[0],high[1],high[2],high[3]);
  }
  else
  {
    std::cout << "Cannot handle weight order "<<low.Order()<<std::endl;
    assert(false);
  }
  return counts;
}

// This implementation is much more general which can handle any weight dim
// But it is unfortunately too slow
// Performance of this function is really critical since will be invoked many time
unsigned NestAnalysis::CountMaskGeneric(problem::DataPoint low, problem::DataPoint high)
{
  unsigned counts=0;
  assert(low.Order()==high.Order());
  for(unsigned int i=0; i<low.Order(); i++)
    assert(low[i]<high[i]);
  const auto& diff = high - low;
  for(unsigned ind=0; ind<diff.Volume(); ind++)
  {
    auto aux = GetAuxiliaryDataPoint(ind, diff);

    const auto& result = low + aux;
    if(GetWeightMaskGeneric(result))
      counts++;
  }
  return counts;
}

/* accessed point must be inclusive */
bool NestAnalysis::GetWeightMask(unsigned dim0, unsigned dim1, unsigned dim2, unsigned dim3)
{
  // do it in a more elegant way?
  assert(shape_.size()==4);

  return weight_snapshots_[sample_id_][dim0*shape_[1]*shape_[2]*shape_[3]+dim1*shape_[2]*shape_[3]+dim2*shape_[3]+dim3];
}

/* accessed point must be inclusive */
bool NestAnalysis::GetWeightMask(unsigned dim0, unsigned dim1, unsigned dim2)
{
  // do it in a more elegant way
  assert(shape_.size()==3);

  return weight_snapshots_[sample_id_][dim0*shape_[1]*shape_[2]+dim1*shape_[2]+dim2];
}

/* The implmentation is more generic which handles arbitrary weight dimension, but the efficiency is under test*/
bool NestAnalysis::GetWeightMaskGeneric(const problem::DataPoint& p)
{
  assert(shape_.size()==p.Order());
  unsigned rem_prod = 1;
  unsigned flattened_weight_index = 0;
  for(int dim = p.Order()-1; dim >= 0; dim--)
  {
    flattened_weight_index += p[dim] * rem_prod;
    rem_prod *= shape_[dim];
  }
  return weight_snapshots_[sample_id_][flattened_weight_index];
}

problem::DataPoint NestAnalysis::GetAuxiliaryDataPoint(unsigned ind, const problem::DataPoint& tile)
{
  problem::DataPoint aux; // auxiliary point similar to (r, s, c, k) in the old design
  unsigned rem_prod = 1; // remaining product, another auxiliary value
  for(int dim = aux.Order()-1; dim >= 0; dim--)
  {
    aux[dim] = ind / rem_prod % tile[dim];
    rem_prod *= tile[dim];
  }
  return aux;
}

void NestAnalysis::AnalysisSparseWorkingSet()
{
  InitAnalysis();
  auto num_storage_levels = storage_tiling_boundaries_.size();
  auto wid = IsDWWU() ? (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : (problem::GetShape()->DataSpaceNameToID.at("Weights"));
  auto working_set_sizes = GetWorkingSetSizes_LTW();

  weights_working_set_.clear(); // otherwise uninitilized then push_back will throw std::back_alloc. TODO: merge to Reset()
  for(auto storage_level : working_set_sizes)
  {
    weights_working_set_.push_back(storage_level[wid]);
  }

  // find out working set we want to analysis
  // ignore full size (e.g. DRAM) and single weight (e.g. MAC Register)
  // also ignore bypassed level
  target_storage_mapping_.clear(); // create mapping between target level and storage level
  for(unsigned i = 0;i<num_storage_levels; i++)
  {
    if((weights_working_set_[i] != 1) && (weights_working_set_[i] != weights_working_set_.back()) && (bypass_nest_.at(wid)[i]))
    { 
      target_storage_mapping_.push_back(i);
    }
    else
    {
      max_utilized_capacity_[i].resize(0);
      cumulative_utilized_capacity_[i].resize(0);
      average_utilized_capacity_[i].resize(0);
      cumulative_overflow_amount_unbalanced_[i].resize(0);
      cumulative_multi_level_overflow_amount_unbalanced_[i].resize(0);
      cumulative_overflow_tiles_unbalanced_[i].resize(0);
      // overflow_freq_[i].resize(0);
      overflow_mask_unbalanced_[i].resize(0);

      // overflow load balancing
      cumulative_overflow_amount_balanced_[i].resize(0);
      cumulative_multi_level_overflow_amount_balanced_[i].resize(0);
      cumulative_overflow_tiles_balanced_[i].resize(0);
      overflow_mask_balanced_[i].resize(0);
    }
  }

  // Important! There exists a possibility that no target level at all (i.e. num_targeted_levels_ can be 0 )
  // An workload that can generate this is a tiny FC
  // When 1) only allocate non weight dim to RF level and
  // 2) GLB and DRAM all equal to full working set 
  auto num_targeted_levels = target_storage_mapping_.size();
  assert(available_weight_budget_.size() >= num_targeted_levels);
  if(num_targeted_levels == 0)
  {
    if(verbose_)
      std::cout <<"no target level found in this mapping, skip sparse ws analysis" << std::endl;
  }
  
  if(verbose_){ 
    for(unsigned int i=0; i<target_storage_mapping_.size();i++)
      std::cout << "Target level "<<i<<" corresponds to storage level "<<target_storage_mapping_[i]<<std::endl; 
    std::cout <<std::endl;
    std::cout <<"available weight budget size: "<<available_weight_budget_.size()<<std::endl;
    std::cout <<"num target levels: "<<num_targeted_levels<<std::endl;
  }

  level_high_.resize(num_targeted_levels); // memoization, no need for DRAM (full)
  memo_spatial_high_.resize(num_targeted_levels); // memoization, no need for DRAM (full)

  problem::DataPoint full;
  full.IncrementAllDimensions();
  full += mold_high_[nest_state_.size()-1].Project(workload_);

  for(int target =num_targeted_levels-1; target>=0;target--) // start from the most outer target level
  {
    // identify base tile
    auto i = target_storage_mapping_[target]; // i is the corresponding storage level, keep it for now for compatibility
    sample_id_=0; // tile analysis is the same for all samples

    problem::DataPoint base, high, spatial_high; // similar to OperationPoint
    high.IncrementAllDimensions(); // exclusive high, high[0] = 1, high[1] = 1;
    auto base_level = storage_tiling_boundaries_[i];

    high += mold_high_[base_level].Project(workload_);

    // memoization
    level_high_[target] = high;

    spatial_high.IncrementAllDimensions();
    // total spatial element to analysis num_spatial_elems_[base_level]
    auto master_level = base_level+1;
    if(spatial_fanouts_[master_level]>1) // master spatial level
    {
      spatial_high += mold_high_[master_level].Project(workload_);
    }
    else if (spatial_fanouts_[master_level]==1) // temporal level
    {
      master_level--;
      spatial_high=high;
    }
    else // non master spatial level
    {
      master_level++;
      while(spatial_fanouts_[master_level]==0) // keep adding if we find non master spatial level
        master_level++;
      spatial_high += mold_high_[master_level].Project(workload_);
    }

    memo_spatial_high_[target] = spatial_high;
    auto num_unique_wws = spatial_high.Volume() / high.Volume();
    auto num_spatial_elems = spatial_fanouts_[master_level];
    weight_multicast_[i] = num_spatial_elems / num_unique_wws;
    
    for(unsigned j=0;j<total_samples_;j++)
    {
      // note here use num_spatial_elems is not accurate since we can have multicast
      // instead, we should use num_unique_wws (number of unique weight working set = number buffers / weight multicast)
      max_utilized_capacity_[i][j].resize(num_unique_wws, 0); // maybe infer from num_spatial_elems_ is easier?
      cumulative_utilized_capacity_[i][j].resize(num_unique_wws, 0);
      average_utilized_capacity_[i][j].resize(num_unique_wws, 0.0);
      cumulative_overflow_amount_unbalanced_[i][j].resize(num_unique_wws, 0);
      cumulative_multi_level_overflow_amount_unbalanced_[i][j].resize(num_unique_wws);
      for(unsigned k=0;k<num_unique_wws;k++)
        cumulative_multi_level_overflow_amount_unbalanced_[i][j][k].resize(num_targeted_levels-target, 0.0);
      cumulative_overflow_tiles_unbalanced_[i][j].resize(num_unique_wws, 0);
      // overflow_freq_[i][j].resize(num_unique_wws, 0.0);
      InitOverflowMask(overflow_mask_unbalanced_[i][j], full/high);

      // overflow load balancing
      cumulative_multi_level_overflow_amount_balanced_[i][j].resize(num_targeted_levels-target, 0.0);
      
      // dingqing FIXME: if we want to implement multilevel regfile, 
      // then overflow mask need to be maintained after we determine how to partition subtile (i.e. divide sub_high instead of high)
      InitOverflowMask(overflow_mask_balanced_[i][j], full/high);
    }


    if(verbose_)
    {
      std::cout << "inferred master level: "<<master_level<<std::endl;
      std::cout << "inferred master fanouts: "<<spatial_fanouts_[master_level]<<std::endl;
      std::cout <<"high: ";
      high.Print();
      std::cout <<"spatial high: ";
      spatial_high.Print();
      std::cout << std::endl;
    }

    // Init extra element Collection
    // This captures the amount of imbalance compared with perfect load balance
    // profiling purpose only
    extra_element_hist_.resize(high.Volume());

    // Evaluate all samples: 
    while(sample_id_<total_samples_)
    {
      if(verbose_)
        std::cout <<"AnalysisSparseWorkingSet sample: "<<sample_id_<<std::endl;
      // iterate spatial working set across full working set
      unsigned num_tiles_per_unique_wws=0;
      // evaluate cycles across spatial PEs
      // dingqing FIXME: How to generally identify the level of level config (like nvdla) for cycle analysis?
      double total_density = 0.0;

      auto spatial_high_repeat = full / spatial_high; // spatial high (chip tile) repeatition at all weight dimention.
      for(unsigned ind = 0; ind < spatial_high_repeat.Volume(); ind++)
      {
        auto aux = GetAuxiliaryDataPoint(ind, spatial_high_repeat);
        base = aux * spatial_high;

        bool dump = false;
        if (dump)
        {
          std::cout<< "base: ";
          base.Print();
        }

        unsigned long max_element = AnalysisSpatialSparseBuffer(i, target, base, high, spatial_high);
        if(verbose_)
          std::cout<<"obtained max working set in a spatial high: "<<max_element<<std::endl;        
        // for perfect LB, all elements in spatial high are evenly distributed.
        auto total_weights_in_spatial_high = CountMask(base, base + spatial_high);
        // Verify CountMaskGeneric implementation, seems proper but essentially too slow
        // assert(CountMaskGeneric(base, base+spatial_high) == CountMask(base, base+spatial_high));
        auto perfect_lb_element = (total_weights_in_spatial_high % num_unique_wws == 0) ?
                                  (total_weights_in_spatial_high / num_unique_wws)      :
                                  (total_weights_in_spatial_high / num_unique_wws) + 1;
        auto max_extra_element_to_process = max_element - perfect_lb_element;
        if(master_level != base_level) // dingqing FIXME: hachy condition (find spatial levels) for evaluate cycles
        {
          total_density += static_cast<double>(max_element) / high.Volume(); 
          extra_element_hist_[max_extra_element_to_process]++;
        }
        num_tiles_per_unique_wws++;
      }
      assert(spatial_high_repeat.Volume() == num_tiles_per_unique_wws);

      // PE cycle modeling:
      // since each weight working set occupies the same amount of cycles in dense case,
      // if PEs operate in an synchronous fastion, then speedup for each spatial high (across buffers)
      // is determined by max density of high (one buffer or one set of multicasted buffers)

      // Notes on Weight Multicasting
      // Note also that spatial_high / high does not necessarily equal to # of spatial elements
      // This implies weights get multicasted. Multicasted weights will take the same amount of cycles
      // to finish executing the assigned operations space. So the above derivation is still valid.
      if(total_density != 0.0) // if we evaluated cycles
      {
        speedup_[sample_id_] = num_tiles_per_unique_wws / total_density; // this speedup implementation consider overflow is cached to the same level
        if(verbose_)
        {
          std::cout << "speedup for sample " << sample_id_ <<" is " << speedup_[sample_id_]<<std::endl;
          std::cout << "ideal speedup for sample " << sample_id_ <<" is " << 1 / sample_sparsity_[sample_id_]<<std::endl;
        }
      }
      // more elegant if we can do c++ equivalent of the following python
      // for au, cu, wws in zip(cumulative_utilized_capacity_, average_utilized_capacity_, weights_working_set_):
      //    au=cu/wws/num_tiles_per_unique_wws
      for(unsigned sid=0; sid<cumulative_utilized_capacity_[i][sample_id_].size(); sid++)
      {
        average_utilized_capacity_[i][sample_id_][sid] = static_cast<double>(cumulative_utilized_capacity_[i][sample_id_][sid])
                                                        / num_tiles_per_unique_wws;
      }

      if(verbose_)
      {
        std::cout <<"For target storage level: "<<i<<std::endl;

        std::cout <<"Maximum Utilized "<<std::endl;
        for(auto mu: max_utilized_capacity_[i][sample_id_])
          std::cout <<mu<< " ";
        std::cout << std::endl;
        std::cout <<"Average Utilized "<<std::endl;
        for(auto au: average_utilized_capacity_[i][sample_id_])
          std::cout <<au<< " ";
        std::cout << std::endl;
        std::cout <<"Overflow Access "<<std::endl;
        for(auto oa: cumulative_overflow_amount_unbalanced_[i][sample_id_])
          std::cout <<oa<< " ";
        std::cout << std::endl;

        unsigned num_unique_wws=max_utilized_capacity_[i][sample_id_].size();
        std::cout <<"num of spatial element: "<<num_unique_wws<<std::endl;
        if(max_utilized_capacity_[i][sample_id_].size()>1){
          std::cout <<"Across All spatial elements: "<<std::endl;
          std::cout <<"Max Maximum Utilized: "
                    <<*std::max_element(max_utilized_capacity_[i][sample_id_].begin(), max_utilized_capacity_[i][sample_id_].end())
                    <<std::endl;
          std::cout <<"Min Maximum Utilized: "
                    <<*std::min_element(max_utilized_capacity_[i][sample_id_].begin(), max_utilized_capacity_[i][sample_id_].end())
                    <<std::endl;
          std::cout <<"Max Average Utilized: "
                    <<*std::max_element(average_utilized_capacity_[i][sample_id_].begin(), average_utilized_capacity_[i][sample_id_].end())
                    <<std::endl;
          std::cout <<"Min Average Utilized: "
                    <<*std::min_element(average_utilized_capacity_[i][sample_id_].begin(), average_utilized_capacity_[i][sample_id_].end())
                    <<std::endl; 
          auto max_coa = *std::max_element(cumulative_overflow_amount_unbalanced_[i][sample_id_].begin(), cumulative_overflow_amount_unbalanced_[i][sample_id_].end());
          auto min_coa = *std::min_element(cumulative_overflow_amount_unbalanced_[i][sample_id_].begin(), cumulative_overflow_amount_unbalanced_[i][sample_id_].end());
          std::cout <<"Max Overflow Accesses: "<<max_coa<<std::endl;
          std::cout <<"Min Overflow Accesses: "<<min_coa<<std::endl;
        }
        else if (max_utilized_capacity_[i][sample_id_].size()==1)
        {
          std::cout <<"Maximum Utilized: "<<max_utilized_capacity_[i][sample_id_][0]<<std::endl;
          std::cout <<"Average Utilized: "<<average_utilized_capacity_[i][sample_id_][0]<<std::endl;
        }
        else
          assert(false);
        std::cout<<std::endl;
      }

      if(verbose_)
      {
        std::cout <<"Overflow Stats: "<<std::endl;
        unsigned long total_overflow_access=0;
        double average_overflow_frequency=0.0;
        auto num_unique_wws = cumulative_overflow_amount_unbalanced_[i][sample_id_].size();
        for(unsigned sid=0; sid<num_unique_wws;sid++)
        {
          total_overflow_access+=cumulative_overflow_amount_unbalanced_[i][sample_id_][sid];
          average_overflow_frequency+=cumulative_overflow_tiles_unbalanced_[i][sample_id_][sid];
        }
        average_overflow_frequency /= num_unique_wws;
        
        std::cout <<"num_tiles_per_unique_wws: "<<num_tiles_per_unique_wws<<std::endl;
        std::cout <<"Total Overflow Accesses: "
                  <<static_cast<double>(total_overflow_access)<<std::endl;
        std::cout <<"Average Per Instance Overflow Access: "
                  <<static_cast<double>(total_overflow_access)/num_unique_wws<<std::endl;

        std::cout <<"Average Overflow Frequency: "<<average_overflow_frequency/num_tiles_per_unique_wws<<std::endl;
        std::cout << std::endl;
      }
      sample_id_++;
    }

    // Profiling: collect Stats for extra elements (how much away from perfect load balance)
    if(verbose_ && master_level != base_level) // dingqing FIXME: this is a hacky way of identifying spatial level)
    {
      std::cout << "Monitor the amount of imbalance compared to perfect load balance: "<<std::endl;
      std::cout << "Execution Overhead resolution: "<<1 / (average_sparsity_ * high.Volume())<<std::endl;
      for(const auto& e: extra_element_hist_)
      {
        std::cout <<e<<", ";
      }
      std::cout << std::endl;
    }
  }

  // gather results
}

void NestAnalysis::InitAnalysis()
{
  speedup_.resize(total_samples_);
  std::fill(speedup_.begin(), speedup_.end(), 1); // speedup defaults to 1
  auto num_storage_levels = storage_tiling_boundaries_.size();
  max_utilized_capacity_.resize(num_storage_levels);
  cumulative_utilized_capacity_.resize(num_storage_levels);
  average_utilized_capacity_.resize(num_storage_levels);
  cumulative_overflow_amount_unbalanced_.resize(num_storage_levels);
  cumulative_multi_level_overflow_amount_unbalanced_.resize(num_storage_levels);
  cumulative_overflow_tiles_unbalanced_.resize(num_storage_levels);
  // overflow_freq_.resize(num_storage_levels);
  overflow_mask_unbalanced_.resize(num_storage_levels);

  // overflow load balancing
  cumulative_overflow_amount_balanced_.resize(num_storage_levels);
  cumulative_multi_level_overflow_amount_balanced_.resize(num_storage_levels);
  cumulative_overflow_tiles_balanced_.resize(num_storage_levels);
  overflow_mask_balanced_.resize(num_storage_levels);

  for(unsigned i = 0; i<num_storage_levels;i++)
  {
    max_utilized_capacity_[i].resize(total_samples_);
    cumulative_utilized_capacity_[i].resize(total_samples_);
    average_utilized_capacity_[i].resize(total_samples_);
    cumulative_overflow_amount_unbalanced_[i].resize(total_samples_);
    cumulative_multi_level_overflow_amount_unbalanced_[i].resize(total_samples_);
    cumulative_overflow_tiles_unbalanced_[i].resize(total_samples_);
    // overflow_freq_[i].resize(total_samples_);
    overflow_mask_unbalanced_[i].resize(total_samples_);

    // overflow load balancing
    cumulative_overflow_amount_balanced_[i].resize(total_samples_, 0);
    cumulative_multi_level_overflow_amount_balanced_[i].resize(total_samples_);
    cumulative_overflow_tiles_balanced_[i].resize(total_samples_, 0);
    overflow_mask_balanced_[i].resize(total_samples_);
  }
  storage_level_tile_access_.resize(num_storage_levels);
  weight_multicast_.resize(num_storage_levels, 1);
}

unsigned long NestAnalysis::AnalysisSpatialSparseBuffer(unsigned storage_level, unsigned target_level,
                                        const problem::DataPoint& spatial_base, 
                                        const problem::DataPoint& mold_high, 
                                        const problem::DataPoint& mold_spatial_high)
{
  // assert relationship between storage level and target level
  // remove after we use storage level for evaluated stats
  assert(target_storage_mapping_[target_level] == storage_level);

  problem::DataPoint base;
  unsigned long current_utilized=0;
  unsigned long max_dense_weight_ws = 0; // max within the current spatial high
  // to clean up, disable max_dense_weight_ws if spatial high == high (i.e. # of spatial element == 1)
  unsigned buffer_id=0; // buffer ID is assigned arbitrarily, later maybe assign it with spatial organization?

  // attack load balancing and overflow by scheduling at finer granularity
  const int num_sub_partition = 2; 
  // to support overflow on subtiles
  typedef std::pair<unsigned,problem::DataPoint> DataPointCountPair; // counted mask and associated sub_low
  std::vector<DataPointCountPair> global_subtile;
  std::vector<std::vector<DataPointCountPair>> cluster_subtile;

  // Disable cluster_subtile first, it's tricky to get this right, re-enable later if necessary
  assert(gGlobalSubtileSorting);
  // std::vector<std::vector<DataPointCountPair>> cluster_subtile; // K cluster, C element per cluster 
  // note for the same k value. psum are spatially reused (since varying C maps to the same psum working set)

  global_subtile.clear();
  cluster_subtile.clear();

  // dingqing FIXME: The following is not general enough to handle replication (e.g. CK both partitioned on a spatial dimension)
  // if we have time, we can work on get this a bit more general

  // Work on WU handling: we will use cnn-layer.cfg instead of weight-update.cfg
  // to reuse code as much as possible
  // The difference is that for any dimension is partitionable (no block constraint) for ideal global sorting
  // With sparse tensor multicast, partition dimension has to align with spatial mapped dimensions
  // example: CN in fw/bw: only partition C, not K
  // example: KN in fw/bw: only partition K, not C
  // in WU: NK CK corresponds to CN, KN as input stationary and output stationary

  // First, figure out if which dimension is spatially mapped onto PE array.
  auto mapped = mold_spatial_high / mold_high;

  // Then, figure out if dimension is actually partitionable
  int par_dim = -1;
  is_partition_ = false;

  std::vector<bool> partitionable;
  partitionable.clear();

  std::vector<unsigned> mapped_dim;
  mapped_dim.clear();
  for(unsigned i=0; i<spatial_base.Order(); i++) // weight order (3 if depthwise, 4 otherwise)
  {
    if(mapped[i] > 1)
      mapped_dim.push_back(i);
    partitionable.push_back(mold_high[i]>1 && mold_high[i]%num_sub_partition==0);
  }
  
  if(verbose_)
  {
    std::cout <<"total mapped dim: "<<mapped_dim.size() <<std::endl;
  }

  if(mapped_dim.size()>2)
  {
    if(verbose_)
      std::cout << "More than 2 mapped weight dim! Something is wrong unless replicated! "
                << "Replication is not properly handled for LB!"
                << std::endl;
  }
  else if(mapped_dim.size()==2)
  {
    if(gGlobalSubtileSorting)
    {
      // both array dimensions are weight related
      // model the ideal network which is hard to build
      // find largest mold_high
      int mapped_largest = 0; // aux only
      for(unsigned i=0;i<partitionable.size(); i++) // iterate all weight dims
      {
        // heuristic that find the largest partitionable dimension to partition
        if(partitionable[i] && mold_high[i]>mapped_largest)
        {
          is_partition_ = !gSyncPE;
          par_dim = i;
          mapped_largest = mold_high[i];
        }
      }
    }
    else // cluster balancing, tricky and currently not maintained
    {
      // This is for CK fw & bw only!
      // The implementation is not general at all!
      if(verbose_)
        std::cout <<"Checking requirement for cluster based balancing" << std::endl;
      assert(problem::GetShape()->name=="CNN-Layer" || problem::GetShape()->name=="Backward-Pass");
      assert(mapped_dim[0]==2 && mapped_dim[1]==3); // very hacky... (can we specify the dim to cluster???)
      // dingding FIXME: try as much as possible to not refer 2 and 3 for C and K, very error prone
      // map K aross clusters and partition C to avoid duplicated RF space
      if(partitionable[2])
      {
        is_partition_ = !gSyncPE;
        par_dim = 2;
        cluster_subtile.resize(mapped[3]);
      }
    }
  }
  else if(mapped_dim.size()==1)
  {
    // only one array dimension is weight related
    // only partition the same dimension
    if(partitionable[mapped_dim[0]])
    {
      is_partition_ = !gSyncPE;
      par_dim = mapped_dim[0];
      // non_par_dim = 3;
      // cluster_subtile.resize(mold_spatial_high[non_par_dim]/mold_high[non_par_dim]);
    }
  }
  else
  {
    // no weight dim on array dim, already optimal balanced
    if(verbose_)
      std::cout << "No weight dim mapped onto array. Optimal already!" << std::endl;
  }

  auto sub_mold_high = mold_high;
  if(is_partition_)
    sub_mold_high[par_dim] /= num_sub_partition;
  sparsity_info_.subtile = sub_mold_high;
  
  // threshold for checking overflow
  unsigned threshold = available_weight_budget_[target_level];

  auto high_repeat = mold_spatial_high / mold_high;

  for(unsigned ind = 0; ind < high_repeat.Volume(); ind++)
  {
    auto aux = GetAuxiliaryDataPoint(ind, high_repeat);
    base = aux * mold_high;
    
    auto low = spatial_base+base;
    auto high = low + mold_high;
    current_utilized=CountMask(low, high);
   
    unsigned sanity_check_count = 0;
    problem::DataPoint sub_base;
    
    if(is_partition_)
    {
      unsigned sub_tile_size = 0;
      for(unsigned i=0; i<num_sub_partition;i++)
      {
        sub_base[par_dim] = i*sub_mold_high[par_dim];
        auto sub_low = low + sub_base;
        auto sub_high = sub_low + sub_mold_high;
        sub_tile_size = CountMask(sub_low, sub_high);
        if(verbose_)
          std::cout <<sub_tile_size<<" ";
        sanity_check_count += sub_tile_size;
        global_subtile.push_back(std::make_pair(sub_tile_size, sub_low));
        if(!cluster_subtile.empty()) // clustered balancing
          cluster_subtile[aux[3]].push_back(std::make_pair(sub_tile_size, sub_low)); // kinda hachy...
      }
      assert(current_utilized==sanity_check_count);
    }
    if(verbose_)
      std::cout <<std::endl;

    if(current_utilized > max_dense_weight_ws) // max across buffer
      max_dense_weight_ws = current_utilized;

    cumulative_utilized_capacity_[storage_level][sample_id_][buffer_id] += current_utilized;
    if(current_utilized > max_utilized_capacity_[storage_level][sample_id_][buffer_id]) // max per buffer
      max_utilized_capacity_[storage_level][sample_id_][buffer_id] = current_utilized;

    if(!is_partition_)
    {
      if(current_utilized>threshold)
      {
        auto overflow_capacity = current_utilized-threshold;
        auto overflow_proportion = static_cast<double>(overflow_capacity)/threshold;

        OverflowAnalysis(target_level,
                         cumulative_overflow_amount_unbalanced_[storage_level][sample_id_][buffer_id],
                         cumulative_overflow_tiles_unbalanced_[storage_level][sample_id_][buffer_id],
                         cumulative_multi_level_overflow_amount_unbalanced_[storage_level][sample_id_][buffer_id],
                         overflow_capacity, 
                         low);
        // To optimized performance, we can disable setting overflow mask for the most inner level
        if(storage_level <= target_storage_mapping_[0]) // most inner target level
          SetOverflowMask(overflow_mask_unbalanced_[storage_level][sample_id_], overflow_proportion, low/mold_high);
      }
    }

    bool dump = false;
    if(dump)
    {
      std::cout<< "base inside spatial tile: ";
      base.Print();
      std::cout<< "CountMask: "<<current_utilized<<std::endl;
      std::cout<< "Cumulated: "<<cumulative_utilized_capacity_[storage_level][sample_id_][buffer_id]<<std::endl;
      std::cout<< "Max element within the same buffer: "<<max_utilized_capacity_[storage_level][sample_id_][buffer_id]<<std::endl;
      std::cout<< "Current max weight ws: "<<max_dense_weight_ws<<std::endl;
    }
    
    buffer_id++; 
    // the variable buffer_id is effectively # of unique weight WS
    // # of buffers = # of unique weight WS * multicast opportunity
  }

  if(verbose_)
  {
	  std::cout<<"gSyncPE: "<<gSyncPE<<std::endl;
	  std::cout<<"is_partition_: "<<is_partition_<<std::endl;
  }

  if(is_partition_)
  {
    unsigned max_combined_tile = 0;
    if(gGlobalSubtileSorting) // for 2 dim sparsity variation, we assume idealized complex NOC that exploits reuse beyond row or column
    {
      // check size
      auto total_subtiles = global_subtile.size();
      assert(total_subtiles == mold_spatial_high.Volume() / mold_high.Volume() * num_sub_partition);
      
      std::sort(global_subtile.begin(), global_subtile.end(), 
        [](auto &left, auto &right) { return left.first > right.first; });

      // even slower solution
      // auto sub_mold_high = mold_high;
      // sub_mold_high[2] /= num_sub_partition;
      // std::sort(global_subtile.begin(), global_subtile.end(), 
      //   [this, &sub_mold_high](auto &left, auto &right) { return CountMask(left, left+sub_mold_high) < CountMask(right, right+sub_mold_high);});
      
      // determine the most dense tile after sorting and folding
      for(unsigned i = 0;i<total_subtiles/2; i++)
      {
        auto combined_tile = global_subtile[i].first+global_subtile[total_subtiles-1-i].first;
        
        // even slower solution
        // auto subtile1 = CountMask(global_subtile[i], global_subtile[i]+sub_mold_high);
        // auto subtile2 = CountMask(global_subtile[total_subtiles-1-i], global_subtile[total_subtiles-1-i]+sub_mold_high);
        // auto combined_tile = subtile1 + subtile2;
        
        if(combined_tile > max_combined_tile)
          max_combined_tile = combined_tile;

        // overflow load balancing
        if(combined_tile>threshold)
        {
          auto overflow_capacity = combined_tile-threshold;
          auto overflow_proportion = static_cast<double>(overflow_capacity)/threshold;
          const auto& sub_low = global_subtile[i].second;

          OverflowAnalysis(target_level,
                           cumulative_overflow_amount_balanced_[storage_level][sample_id_],
                           cumulative_overflow_tiles_balanced_[storage_level][sample_id_],
                           cumulative_multi_level_overflow_amount_balanced_[storage_level][sample_id_],
                           overflow_capacity, 
                           sub_low);
          // To optimized performance, we can disable setting overflow mask for the most inner level
          if(storage_level <= target_storage_mapping_[0]) // most inner target level
          {
            // we should change mold_high to sub_mold_high if we have multilevel regfile
            // Now if sub tile overflows, we mark the entire origin tile as overflow
            SetOverflowMask(overflow_mask_balanced_[storage_level][sample_id_], overflow_proportion, sub_low/mold_high);
          }
        }
      }
    }
    else // cluster subtile sorting, not maintained currently
    {
      auto num_clusters = cluster_subtile.size();
      for(unsigned i = 0; i < num_clusters; i++)
      {
        auto& cluster = cluster_subtile[i];
        // check size
        auto subtile_per_cluster = cluster.size();
        assert(static_cast<std::int32_t>(subtile_per_cluster) == mold_spatial_high[par_dim] / mold_high[par_dim] * num_sub_partition); // spatial C
        
        if(verbose_)
        {
          std::cout<<"sub-tiles before sorting"<<std::endl;
          for(unsigned j = 0; j<cluster.size(); j++){
            std::cout<<cluster[j].first<<" ";
          }
          std::cout<<std::endl;
        }
        // expensive check on sorted results:
        // auto sub_mold_high = mold_high;
        // sub_mold_high[2] /= num_sub_partition;
        // for(unsigned i = 0; i<subtile_per_cluster;i++)
        //   assert(cluster[i].first == CountMask(cluster[i].second,cluster[i].second + sub_mold_high));
        std::sort(cluster.begin(), cluster.end(),
          [](auto &left, auto &right) { return left.first > right.first; });

        if(verbose_)
        {
          std::cout<<"sub-tiles after sorting"<<std::endl;
          for(unsigned j = 0; j<cluster.size(); j++){
            std::cout<<cluster[j].first<<" ";
          }
          std::cout<<std::endl;
        }
        // even slower solution
        // auto sub_mold_high = mold_high;
        // sub_mold_high[2] /= num_sub_partition;
        // std::sort(global_subtile.begin(), global_subtile.end(), 
        //   [this, &sub_mold_high](auto &left, auto &right) { return CountMask(left, left+sub_mold_high) < CountMask(right, right+sub_mold_high);});
        
        for(unsigned j = 0; j < subtile_per_cluster/2; j++)
        {
          auto combined_tile = cluster[j].first + cluster[subtile_per_cluster-1-j].first;
          
          // even slower solution            
          // auto subtile1 = CountMask(cluster[i], cluster[i]+sub_mold_high);
          // auto subtile2 = CountMask(cluster[subtile_per_cluster-1-i], cluster[subtile_per_cluster-1-i]+sub_mold_high);
          // auto combined_tile = subtile1 + subtile2;
          
          if(combined_tile > max_combined_tile)
            max_combined_tile = combined_tile;

          // overflow load balancing
          if(combined_tile>threshold)
          {
            auto overflow_capacity = combined_tile-threshold;
            auto overflow_proportion = static_cast<double>(overflow_capacity)/threshold;
            const auto& sub_low = cluster[j].second;

            OverflowAnalysis(target_level,
                             cumulative_overflow_amount_balanced_[storage_level][sample_id_],
                             cumulative_overflow_tiles_balanced_[storage_level][sample_id_],
                             cumulative_multi_level_overflow_amount_balanced_[storage_level][sample_id_],
                             overflow_capacity, 
                             sub_low);
            // To optimized performance, we can disable setting overflow mask for the most inner level
            if(storage_level <= target_storage_mapping_[0]) // most inner target level
            {
              // we should change mold_high to sub_mold_high if we have multilevel regfile
              // Now if sub tile overflows, we mark the entire origin tile as overflow
              SetOverflowMask(overflow_mask_balanced_[storage_level][sample_id_], overflow_proportion, sub_low/mold_high);
            }
          }
        }
      }
    }
    // std::cout << "The most dense load balanced wws contains "<<max_combined_tile<<" elements"<<std::endl;
    return max_combined_tile;
  }

  return max_dense_weight_ws;
}

void NestAnalysis::OverflowAnalysis(unsigned target_level,
                                    unsigned long& cumulative_access,
                                    unsigned long& cumulative_tiles,
                                    MultiLevelOverflow& multi_level_access,
                                    unsigned long overflow_capacity,
                                    problem::DataPoint base)
{
  auto total_overflow_access = overflow_capacity; // we multiply storage_level_tile_access_[target_level] when collecting data
  cumulative_access += total_overflow_access;
  cumulative_tiles++;

  // recursively check if upper level also overflows (up to num_targeted_levels-1)
  double unallocated = static_cast<double>(total_overflow_access);
  for(unsigned i=0;i<multi_level_access.size();i++)
  {
    if(i == multi_level_access.size()-1) // next non-bypass level contains full WS
    {
      auto allocated = unallocated;
      multi_level_access[i] += allocated;
      unallocated -= allocated;
    }
    else // next non-bypass level is actually the next target level (this is not true for all target level)
    {
      // loop up next level overflow mask, and determine to which next level tile it belongs
      auto next_target_level = target_level+i+1;
      assert(next_target_level < target_storage_mapping_.size());
      double overflow_proportion = 0.0;
      if(weights_working_set_[target_storage_mapping_[next_target_level]]!=weights_working_set_.back())
        overflow_proportion = GetOverflowMask(overflow_mask_unbalanced_[target_storage_mapping_[next_target_level]][sample_id_], base/level_high_[next_target_level]);
      
      if(overflow_proportion==0)
      {
        // no overflow: allocate the rest and break

        auto allocated = unallocated;
        multi_level_access[i] += allocated;
        unallocated -= allocated;
        break;
      }
      else
      {
        // allocate current level
        auto allocated = (1.0-overflow_proportion)*unallocated;
        multi_level_access[i] += allocated;
        unallocated -= allocated;
      }
    }
  }

  if(unallocated!=0)
  {
    std::cout << "failed mapping id: " << mapping_id_ <<std::endl;
    std::cout <<"unallocated: "<<unallocated<<std::endl;
  }

  if((std::accumulate(multi_level_access.begin(), multi_level_access.end(), static_cast<double>(0)) 
          - static_cast<double>(cumulative_access))>1e-6)
  {
    std::cout << "failed mapping id:" << mapping_id_ << std::endl;
    std::cout << "cumulative access:" << cumulative_access<<std::endl;
    std::cout << "total multi_level_access:" << std::accumulate(multi_level_access.begin(), multi_level_access.end(), static_cast<double>(0)) <<std::endl;
  }

  assert((unallocated - 0.0)<1e-6);
  assert((std::accumulate(multi_level_access.begin(), multi_level_access.end(), static_cast<double>(0)) 
          - static_cast<double>(cumulative_access))<1e-6);
}

void NestAnalysis::InitOverflowMask(OverflowMask& mask, problem::DataPoint p)
{
  assert(p.Order() == 3 || p.Order() == 4);
  mask.shape = p;
  mask.data.resize(p.Volume(), 0.0);
}

double NestAnalysis::GetOverflowMask(OverflowMask& mask, problem::DataPoint p)
{
  if(p.Order() == 3)
    return mask.data[p[0]*mask.shape[1]*mask.shape[2]+p[1]*mask.shape[2]+p[2]];
  else if(p.Order() == 4)
    return mask.data[p[0]*mask.shape[1]*mask.shape[2]*mask.shape[3]+p[1]*mask.shape[2]*mask.shape[3]+p[2]*mask.shape[3]+p[3]];
  else
    {
      std::cout << "Cannot handle weight order " << p.Order() << std::endl;
      assert(false);
    }
}

void NestAnalysis::SetOverflowMask(OverflowMask& mask, double value, problem::DataPoint p)
{
  if(p.Order() == 3)
     mask.data[p[0]*mask.shape[1]*mask.shape[2]+p[1]*mask.shape[2]+p[2]] = value;
  else if(p.Order() == 4)
    mask.data[p[0]*mask.shape[1]*mask.shape[2]*mask.shape[3]+p[1]*mask.shape[2]*mask.shape[3]+p[2]*mask.shape[3]+p[3]] = value;
  else
    {
      std::cout << "Cannot handle weight order " << p.Order() << std::endl;
      assert(false);
    }
}

void NestAnalysis::DisplayOverflowMask(OverflowMask& mask)
{
  for(unsigned i = 0; i <mask.data.size();i++)
    std::cout << mask.data[i] << " ";
  std::cout << std::endl;
}

unsigned NestAnalysis::GetWeightOrder()
{
  if(workload_->GetShape()->name == "Weight-Update-Depthwise")
    return workload_->GetShape()->DataSpaceOrder.at(workload_->GetShape()->DataSpaceNameToID.at("Inputs"));
  else
    return workload_->GetShape()->DataSpaceOrder.at(workload_->GetShape()->DataSpaceNameToID.at("Weights"));
}

void NestAnalysis::SortingCost()
{ 
  if(is_partition_)
  {
    // Evaluate swaps
    const auto &reg_wws = level_high_[0]; // dingqing FIXME: Assumption that take target level 0 is not general considering say multi regfile

    // based on fw/bw to figure out hw para
    const auto &sorting_wws = memo_spatial_high_.at(0); // for bound checking

    auto num_subtiles = 2 * sorting_wws.Volume()/reg_wws.Volume();
    double num_swaps = num_subtiles * std::log2(num_subtiles);
    auto num_sorting_wws = weights_working_set_.back() / sorting_wws.Volume();
    auto num_swaps_full_ws = num_sorting_wws * num_swaps;
    
    if(verbose_)
    {
      std::cout <<"num_subtiles: " << num_subtiles << std::endl;
      std::cout <<"num_sorting_wws: " << num_sorting_wws << std::endl;
    }
    // Evaluate blk_ptr_access
    // Within sorting_wws
    
    auto total_subtiles = num_sorting_wws * num_subtiles;

    auto ofid = problem::GetShape()->DataSpaceNameToID.at("Outputs");
    auto ifid = problem::GetShape()->DataSpaceNameToID.at("Inputs");
    unsigned long blk_ptr_access = 0;
    bool is_aligned = false;
    if (problem::GetShape()->IsReadWriteDataSpace.at(ofid)) // forward pass
    {
      is_aligned = true;
      blk_ptr_access = 2; // best case
    }
    else if (problem::GetShape()->IsReadWriteDataSpace.at(ifid)) // backward pass
    {
      // auto num_blk_per_subtile = reg_wws.Volume() / (shape_[0] * shape_[1]) / 2;
      // blk_ptr_access = 2 * num_blk_per_subtile; // worse case
      // Since we can reorder weights during GLB filling, we can use end - start, therefore 2 blk ptr access
      is_aligned = true;
      blk_ptr_access = 2; // best case
    }

    auto blk_ptr_access_full_ws = total_subtiles * blk_ptr_access;
    if(verbose_)
    {
      std::cout <<"total swaps in the full working set: " << num_swaps_full_ws << std::endl;
      std::cout <<"total subtiles: "<<total_subtiles<<std::endl;
      std::cout << "Weight aligned"<<is_aligned<<std::endl;
      std::cout <<"total blk ptr acc in the full working set: " << blk_ptr_access_full_ws << std::endl;
    }
    // The above results need to be scaled with storage_level_tile_access_ at glb level
    // in CollectWorkingSets()
    sparsity_info_.swaps = std::ceil(num_swaps_full_ws) * storage_level_tile_access_[1];
    sparsity_info_.blk_ptr_access = blk_ptr_access_full_ws * storage_level_tile_access_[1];
  }
}

void NestAnalysis::CollectWorkingSets()
{
  // Collect the data we want to return. Transpose the max_size_ and accesses_
  // matrix, pack them into an array of vectors and return.
  for (auto& cur : nest_state_)
  {
    // All spatial levels that are not a master-spatial level are not valid
    bool valid_level = !loop::IsSpatial(cur.descriptor.spacetime_dimension) || master_spatial_level_[cur.level];
    if (valid_level)
    {
      // Contains the collected state for this level.
      analysis::ElementState condensed_state;
      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        // Sanity check: All elements in a given level should
        // have similar working sets, accesses etc.
        // TODO Can we leverage this assertion to avoid redundant simulation
        // by only simulating one spatial element per level?
        if (!gExtrapolateUniformSpatial)
        {
          for (std::uint64_t i = 1; i < cur.live_state.size(); i++)
          {
            ASSERT(cur.live_state[i].accesses[pv] ==
                   cur.live_state[i - 1].accesses[pv]);
            ASSERT(cur.live_state[i].max_size[pv] ==
                   cur.live_state[i - 1].max_size[pv]);
            ASSERT(cur.live_state[i].link_transfers[pv] ==
                   cur.live_state[i - 1].link_transfers[pv]);
          }
        }

        // Since, all elements have the same properties, use the properties
        // of the first element to build condensed_state
        const uint64_t REPR_ELEM_ID = 0;  // representative element id.
        condensed_state.accesses[pv] =
            cur.live_state[REPR_ELEM_ID].accesses[pv];
        condensed_state.scatter_factors[pv] =
            cur.live_state[REPR_ELEM_ID].scatter_factors[pv];
        condensed_state.cumulative_hops[pv] =
            cur.live_state[REPR_ELEM_ID].cumulative_hops[pv];
        condensed_state.max_size[pv] =
            cur.live_state[REPR_ELEM_ID].max_size[pv];
        condensed_state.link_transfers[pv] =
            cur.live_state[REPR_ELEM_ID].link_transfers[pv];
      }

      // Build the subnest corresponding to this level.
      // We need a vector of nests because a master spatial level's
      // subnest should include the nests of the slave spatial levels.
      // This is very useful for debugging purposes.
      std::vector<loop::Descriptor> subnest;
      subnest.push_back(cur.descriptor);
      if (master_spatial_level_[cur.level])
      {
        int l = cur.level - 1;
        while (l >= 0 && loop::IsSpatial(nest_state_[l].descriptor.spacetime_dimension))
        {
          subnest.push_back(nest_state_[l].descriptor);
          l--;
        }
        std::reverse(subnest.begin(), subnest.end());
      }

      // Transfer data from condensed_state to working_sets_
      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        tiling::TileInfo tile;
        tile.size                   = condensed_state.max_size[pv];
        tile.partition_size         = 0; // will be set later.
        tile.accesses               = condensed_state.accesses[pv]; // network accesses
        tile.fills                  = 0; // will be set later
        tile.scatter_factors        = condensed_state.scatter_factors[pv];
        tile.cumulative_hops        = condensed_state.cumulative_hops[pv];
        tile.content_accesses       = tile.GetTotalAccesses();
        tile.link_transfers         = condensed_state.link_transfers[pv];
        tile.subnest                = subnest;
        tile.replication_factor     = num_spatial_elems_[cur.level];
        tile.fanout                 = spatial_fanouts_[cur.level];
        tile.is_on_storage_boundary = storage_boundary_level_[cur.level];
        tile.is_master_spatial      = master_spatial_level_[cur.level];
        working_sets_[pv].push_back(tile);
      }

    } // if (valid_level)
  } // for (nest)

  // Extract body data from innermost spatial level.
  for (auto& cur : nest_state_)
  {
    // All spatial levels that are not a master-spatial level are not valid
    bool valid_level = !loop::IsSpatial(cur.descriptor.spacetime_dimension) || master_spatial_level_[cur.level];
    if (valid_level)
    {
      body_info_.replication_factor = num_spatial_elems_[cur.level] * spatial_fanouts_[cur.level];
      break;
    }
  }

  // total speedup
  double tmp_accumulator = 0.0;
  for(const auto& s : speedup_) // across different snapshots
    tmp_accumulator += 1/s; // exe time determined by accumulation of sparsity
  auto total_speedup = speedup_.size() / tmp_accumulator;
  auto total_ideal_speedup = 1/average_sparsity_;
  
  if(verbose_)
  {
    std::cout << "Total speedup: "<<total_speedup<<std::endl;
    std::cout << "Ideal speedup: "<<total_ideal_speedup<<std::endl;
  }
  
  // collapse early and use it to evaluate storage_level_tile_access_
  auto num_storage_levels = storage_tiling_boundaries_.size();
  std::vector<tiling::TileInfo> collapsed_weight_access; 
  collapsed_weight_access.clear();
  auto wid = IsDWWU() ? (problem::GetShape()->DataSpaceNameToID.at("Inputs")) : (problem::GetShape()->DataSpaceNameToID.at("Weights"));  
  tiling::Collapsing(working_sets_, num_storage_levels, wid, collapsed_weight_access); // collapsing only
  for(unsigned i = 0; i <num_storage_levels; i++)
  {
    unsigned num_spatial_elem=max_utilized_capacity_[i][0].size() * weight_multicast_[i]; // storage level, arbitrary sample
    if(num_spatial_elem == 0) // non target level
      num_spatial_elem = 1;

    storage_level_tile_access_[i] = collapsed_weight_access[i].content_accesses * num_spatial_elem / weights_working_set_.back();
    if(verbose_)
      std::cout << "storage_level_tile_access_ at level "<<i<<" : "<<storage_level_tile_access_[i]<<std::endl;
  }

  // collect sparsity related info
  // weight_sparsity is used for scaled access count
  // buffer size is used for scaled cost per access (useful when partitioned buffer supported)
  // overflow is used for evaluating overhead when WS doesn't fit in the analytical buffer (partitioned weight buffer)
  sparsity_info_.weight_sparsity = average_sparsity_;
  sparsity_info_.inverse_speedup = 1/total_speedup;

  sparsity_info_.overflow.resize(num_storage_levels, 0.0);
  sparsity_info_.unfilled.resize(num_storage_levels, 0.0);
  sparsity_info_.overflow_breakdown.resize(num_storage_levels);

  unsigned target = 0;
  for(unsigned i=0; i<num_storage_levels; i++)
  {
    // vector does not have support for bound check, annoying
    // we can get rid of this by iterate over target levels
    if(target<target_storage_mapping_.size() && target_storage_mapping_[target]==i) // if the storage level is a target level
    {
      unsigned long count=0;
      std::vector<double> breakdown_count;
      unsigned long lb_count=0;
      std::vector<double> lb_breakdown_count;
      auto num_unique_wws = cumulative_overflow_amount_unbalanced_[i][0].size();
      auto num_breakdown_levels = cumulative_multi_level_overflow_amount_unbalanced_[i][0][0].size();
      breakdown_count.resize(num_breakdown_levels, 0.0);
      for(unsigned j=0;j<total_samples_;j++)
      {
        for(unsigned k=0;k<num_unique_wws;k++) // spatial element
        {
          count += cumulative_overflow_amount_unbalanced_[i][j][k];
          for(unsigned l=0;l<num_breakdown_levels;l++)
          {
            breakdown_count[l] += cumulative_multi_level_overflow_amount_unbalanced_[i][j][k][l];
          }
        }
      }

      // overflow load balancing
      lb_breakdown_count.resize(num_breakdown_levels, 0.0);
      for(unsigned j=0;j<total_samples_;j++)
      {
        lb_count += cumulative_overflow_amount_balanced_[i][j];
        for(unsigned k=0;k<num_breakdown_levels;k++)
        {
          lb_breakdown_count[k] += cumulative_multi_level_overflow_amount_balanced_[i][j][k];
        }
      }

      // fills are determined by the next upper non-bypassed storage level
      auto next_non_bypass_level = i+1;
      while(!bypass_nest_[wid].test(next_non_bypass_level))
        next_non_bypass_level++;
      sparsity_info_.overflow_breakdown[i].resize(num_breakdown_levels, 0.0);

      // sync PE
      if(gSyncPE)
      {
        sparsity_info_.overflow[i] = static_cast<double>(count)*storage_level_tile_access_[i]/total_samples_;
        sparsity_info_.unfilled[i] = static_cast<double>(count)*storage_level_tile_access_[next_non_bypass_level]/total_samples_; 
        std::transform(breakdown_count.begin(), breakdown_count.end(),
                       sparsity_info_.overflow_breakdown[i].begin(), 
                       [i, this](double c){return c*storage_level_tile_access_[i]/total_samples_;}); // need to capture this for member total_samples_
      }
      else // load balanced PE
      {
        sparsity_info_.overflow[i] = static_cast<double>(lb_count)*storage_level_tile_access_[i]/total_samples_;
        sparsity_info_.unfilled[i] = static_cast<double>(lb_count)*storage_level_tile_access_[next_non_bypass_level]/total_samples_; 
        std::transform(lb_breakdown_count.begin(), lb_breakdown_count.end(),
                       sparsity_info_.overflow_breakdown[i].begin(), 
                       [i, this](double c){return c*storage_level_tile_access_[i]/total_samples_;}); 
      }

      target++;
    }
    else if (weights_working_set_[i] == weights_working_set_.back())
    {
      sparsity_info_.overflow_breakdown[i].resize(0);
    }
    else if (weights_working_set_[i] == 1)
    {
      // trivial case that we have a register at MAC... dingqing FIXME
      sparsity_info_.overflow[i]=0; // FIXME
      sparsity_info_.unfilled[i]=0; // FIXME
      sparsity_info_.overflow_breakdown[i].resize(0); // FIXME
    }
    else // level that masked off with bypass nest, do nothing (clear)
    {
      // all default to 0
      sparsity_info_.overflow_breakdown[i].resize(0);
    }  
  }
  assert(target == target_storage_mapping_.size());
  
  // dingqing FIXME: glb levels is not necessary 1 if bypassed/multi-level regfile. Make it more general later
  // This function should be skipped if no target level
  // We can reorganize the blocks to reduce the sorting cost by only subtracking head/tail block ptr
  // instead of accessing all relevant block ptrs
  if(target_storage_mapping_.size() > 0)
    SortingCost();

  if(verbose_)
  {
    std::cout <<"Collect Sparsity related info: "<<std::endl;
    sparsity_info_.Print();
  }
}

// Delta computation (recursive call).
// Unless skip_delta is true, returns the delta between the working set of the
// previous iteration and the current iteration of the current level.
problem::OperationSpace NestAnalysis::ComputeDeltas(
    std::vector<analysis::LoopState>::reverse_iterator cur, bool skip_delta)
{
  ASSERT(cur != nest_state_.rend());
  ASSERT(spatial_id_ < cur->live_state.size());

  if (gTerminateEval)
  {
    throw std::runtime_error("terminated");
  }

  // cur->live_state is a vector of ElementState with size: # of spatial elements
  auto& cur_state = cur->live_state[spatial_id_];
  
  // The point set for this invocation. Note that we do *not* initialize this to
  // the last-seen state at the end of the prior invocation. Doing so causes the
  // state at this level to grow indefinitely, which isn't what we're trying to
  // model. The responsibility of this level is to supply all the deltas
  // demanded by the next-inner level for this invocation.
  problem::OperationSpace point_set(workload_);

  if (loop::IsSpatial(cur->descriptor.spacetime_dimension))
  {
    ComputeSpatialWorkingSet(cur, point_set);
  }
  else
  {
    ComputeTemporalWorkingSet(cur, point_set, cur_state);
  }

  int level = cur->level;

  // Record the maximum point set size ever seen across all invocations
  // of this level.
  // Need to be done only for levels which will map to physical storage levels
  // after we run collapseTiles.
  // Also need to do this for master spatial levels in order to calculate
  // partition size later.
  if (storage_boundary_level_[level] || master_spatial_level_[level])
  {
    auto sizes = point_set.GetSizes(); // return size for each dataspace
    // OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2, 
    //                 OutputIt d_first, BinaryOperation binary_op );
    std::transform(sizes.begin(), sizes.end(), cur_state.max_size.begin(),
                   cur_state.max_size.begin(),
                   [](std::size_t x, std::size_t y) { return std::max(x, y); });
  }

  // Reset indices
  indices_[level] = cur->descriptor.start;

  bool dump = false; // (level >= 4);
  if (dump)
  {
    std::cout << "--------------------\n";
    std::cout << "LEVEL " << level << " (DeltaCalc)" << std::endl;
    std::cout << "--------------------\n";

    std::cout << "Last:\n    ";
    cur_state.last_point_set.Print();
    std::cout << "New:\n    ";
    point_set.Print();
  }
  
  // Calculate delta to send up to caller.
  problem::OperationSpace delta(workload_);
  if (!skip_delta)
  {
    delta = point_set - cur_state.last_point_set;
  }

  if (dump)
  {
    std::cout << "Delta:\n    ";
    delta.Print();
    //std::cout << "    New after Minus op:\n";
    //point_set.Print();
  }    

  // Update last-seen point set for this level.
  cur_state.last_point_set = point_set;

  return delta;
}

void NestAnalysis::ComputeTemporalWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                     problem::OperationSpace& point_set,
                                     analysis::ElementState& cur_state)
{
  // We do two things in this function: (a) calculate the size of the temporal
  // working set for this level, and (b) calculate the number of accesses to
  // this level from the inner level.
  //
  // We used to do both these tasks by accumulating the deltas returned by
  // recursive calls to inner nesting levels. That was problematic for task (a)
  // because inner levels can sometimes buffer data across iterations of *this*
  // level, which sometimes causes the union of the deltas propagated to this
  // level to form a fractured polyhedral space. Note that this fracturing is
  // fine in terms of calculating *accesses* to this level (b), since it
  // reflects filtering.
  //
  // To address this, we first attempted to restrict gradient direction changes
  // during delta computation. However, this only captures a subset of scenarios.
  // It also affects (b), but that is fine because most hardware pattern
  // generators are probably going to be unable to generate patterns that can
  // keep state alive through gradient direction changes.
  //
  // The solution we are now adopting is to use delta accumulation only for (b)
  // and to use an alternative tactic for (a). For certain problem shapes (such
  // as CNN's axis-aligned hyper-rectangles), we can trivially calculate working
  // sets by considering only the corner points in the problem sub-space walked
  // by the subnest starting from this level down. We assume that loops are
  // always ascending (FIXME: check for this during loop construction).

  int level = cur->level;

  bool dump = false; // (level >= 4);
  
  //
  // Step I: Compute Temporal Working Set.
  //

  problem::OperationPoint low_problem_point;
  problem::OperationPoint high_problem_point;

  // We use the pre-computed molds within this level range.
  // Above this level range, we use the transform problem-point to
  // translate, rotate or otherwise transform the mold.
  for (unsigned dim = 0; dim < unsigned(problem::GetShape()->NumDimensions); dim++)
  {
    low_problem_point[dim] = cur_transform_[dim] + mold_low_[level][dim];
    high_problem_point[dim] = cur_transform_[dim] + mold_high_[level][dim];
  }

  // Compute the polyhedron between the low and high problem
  // points (exclusive). Note that this special constructor
  // is only available for certain point-set implementations.
  point_set = problem::OperationSpace(workload_, low_problem_point, high_problem_point); // += previously

  if (dump)
  {
    std::cout << "Final point set:\n    ";
    point_set.Print();
  }

  //
  // Step II: Compute Accesses by accumulating deltas returned by inner levels.
  //
  std::uint64_t num_iterations = 1 +
    ((cur->descriptor.end - 1 - cur->descriptor.start) /
     cur->descriptor.stride);

  if (level == 0) // base
  {
    auto body_iterations = num_iterations * num_epochs_;
    // macs_ += body_iterations;
    if (spatial_id_ == 0)
    {
      // To avoid double counting of compute_cycles when there are multiple PEs.
      // compute_cycles_ += body_iterations;
      body_info_.accesses += body_iterations;
    }

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      // Write-backs of read-modify-write data types consume 2
      // accesses *except* for the first write.
      if (problem::GetShape()->IsReadWriteDataSpace.at(pv) &&
          cur_state.accesses[pv][0] != 0)
      {
        cur_state.accesses[pv][0] += body_iterations; // (2 * body_iterations); This fixup now happens in model/buffer.cpp.
      }
      else
      {
        cur_state.accesses[pv][0] += body_iterations;
      }

      // Set scatter factor (otherwise it will stay at 0 for temporal levels).
      cur_state.scatter_factors[pv][0] = 1;

      // Set cumulative hops for temporal levels.
      cur_state.cumulative_hops[pv][0] = 0.0;
    }
  }
  else // recurse
  {
    std::vector<problem::PerDataSpace<std::size_t>> temporal_delta_sizes;
    std::vector<std::uint64_t> temporal_delta_scale;

    bool run_last_iteration = false;
      
    if (gExtrapolateUniformTemporal)
    {
      // What we would like to do is to *NOT* iterate through the entire loop
      // for this level, but instead fire iterations #0, #1 and #last, and
      // extrapolate the remainder based on the result of iteration #1.

      // Iteration #last is only required for accurate partition size tracking.
      // Otherwise, we reset the point set on any gradient change, and so
      // tracking the point set for the #last iteration is not needed.

      // Note that this entire approach will break if there is any irregularity
      // in working-set movement along the loop (e.g., a modulus in the index
      // expression).

      int dim = int(cur->descriptor.dimension);
      int scale = per_level_dim_scales_[level][dim]; // use to infer operation space tile within this level
      auto saved_transform = cur_transform_[dim];

      // Iteration #0.
      indices_[level] = cur->descriptor.start;
      if (num_iterations >= 1)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeDeltas(cur, false);
        --cur;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(1);
        cur_transform_[dim] += scale; // per_level_dim_scales_[level][dim]

        indices_[level] += cur->descriptor.stride;
      }

      // Iterations #1 through #last-1/last.
      if ((run_last_iteration && num_iterations >= 3) ||
          (!run_last_iteration && num_iterations >= 2))
      {
        // Invoke next (inner) loop level, scaling up the number of epochs
        // by the number of virtual iterations we want to simulate.
        std::uint64_t virtual_iterations =
          run_last_iteration ? num_iterations - 2 : num_iterations - 1;

        auto saved_epochs = num_epochs_;
        num_epochs_ *= virtual_iterations;

        ++cur;
        auto temporal_delta = ComputeDeltas(cur, false);
        --cur;

        num_epochs_ = saved_epochs;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(virtual_iterations);

        cur_transform_[dim] += (scale * virtual_iterations);

        indices_[level] += (cur->descriptor.stride * virtual_iterations);
      }

      // Iteration #last.
      if (run_last_iteration && num_iterations >= 2)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeDeltas(cur, false);
        --cur;

        // If we ran the virtual-iteration logic above, we shouldn't actually
        // use this returned delta, because we will receive the delta between
        // iteration #2 and #last. Instead, we just re-use the last delta by
        // increasing the #virtual iterations (scale) by 1.
        if (num_iterations >= 3)
        {
          temporal_delta_scale.back()++;
        }
        else
        {
          temporal_delta_sizes.push_back(temporal_delta.GetSizes());
          temporal_delta_scale.push_back(1);
          cur_transform_[dim] += scale;
        }
      
        indices_[level] += cur->descriptor.stride;        
      }

      cur_transform_[dim] = saved_transform;
    }
    else // not gExtrapolateUniformTemporal
    {
      int dim = int(cur->descriptor.dimension);
      int scale = per_level_dim_scales_[level][dim];

      auto saved_transform = cur_transform_[dim];

      for (indices_[level] = cur->descriptor.start;
           indices_[level] < cur->descriptor.end;
           indices_[level] += cur->descriptor.stride)
      {
        // Invoke next (inner) loop level.
        ++cur;
        auto temporal_delta = ComputeDeltas(cur);
        --cur;

        temporal_delta_sizes.push_back(temporal_delta.GetSizes());
        temporal_delta_scale.push_back(1);

        cur_transform_[dim] += scale;
      }

      cur_transform_[dim] = saved_transform;
    } // gExtrapolateUniformTemporal
    
    if (dump)
    {
      std::cout << "-------\n";
      std::cout << "LEVEL " << level << std::endl;
      std::cout << "-------\n";
    }

    if (storage_boundary_level_[level - 1])
    {
      // Track accesses for only those levels that are relevant
      // in the final analysis after CollapseTiles.
      problem::PerDataSpace<std::size_t> final_delta_sizes;
      final_delta_sizes.fill(0);

      auto num_deltas = temporal_delta_sizes.size();
      for (unsigned i = 0; i < num_deltas; i++)
      {
        for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
        {
          final_delta_sizes[pv] += (temporal_delta_sizes[i][pv] * temporal_delta_scale[i]);
        }
      }

      for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
      {
        // Write-backs of read-modify-write data types consume 2
        // accesses *except* for the first write.
        if (problem::GetShape()->IsReadWriteDataSpace.at(pv) &&
            cur_state.accesses[pv][0] != 0)
        {
          cur_state.accesses[pv][0] += final_delta_sizes[pv] * num_epochs_; // (2 * final_delta_sizes[pv] * num_epochs_); This fixup now happens in model/buffer.cpp.
        }
        else
        {
          cur_state.accesses[pv][0] += final_delta_sizes[pv] * num_epochs_;
        }

        // Set scatter factor (otherwise it will stay at 0 for temporal levels).
        cur_state.scatter_factors[pv][0] = 1;

        // Set cumulative hops for temporal levels.
        cur_state.cumulative_hops[pv][0] = 0.0;

        // Update delta histogram. Hypothesis is we only need to do this for temporal levels.
        cur_state.delta_histograms[pv][final_delta_sizes[pv]] += num_epochs_;
        
      } // for (datatype)
    } // storage boundary
    
  } // level > 0
}

void NestAnalysis::ComputeSpatialWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                            problem::OperationSpace& point_set)
{
  int level = cur->level;
  ASSERT(master_spatial_level_[level]);

  //
  // Step I - Compute Spatial Working Set.
  //

  problem::OperationPoint low_problem_point;
  problem::OperationPoint high_problem_point;

  // We use the pre-computed molds within this level range.
  // Above this level range, we use the transform problem-point to
  // translate, rotate or otherwise transform the mold.
  for (unsigned dim = 0; dim < unsigned(problem::GetShape()->NumDimensions); dim++)
  {
    low_problem_point[dim] = cur_transform_[dim] + mold_low_[level][dim];
    high_problem_point[dim] = cur_transform_[dim] + mold_high_[level][dim];
  }

  // Compute the polyhedron between the low and high problem
  // points (exclusive). Note that this special constructor
  // is only available for certain point-set implementations.
  // Note: we aren't using +=. This means we're ignoring subvolumes
  // returned to us by recursive FillSpatialDeltas calls.
  point_set = problem::OperationSpace(workload_, low_problem_point, high_problem_point);

  //
  // Step II: Compute Spatial Deltas, etc.
  //

  std::uint64_t num_spatial_elems = spatial_fanouts_[level];
  spatial_id_ *= num_spatial_elems;

  // Deltas needed by each of the spatial elements.
  // This array will be filled by recursive calls.
  std::vector<problem::OperationSpace> spatial_deltas(num_spatial_elems,
                                                    problem::OperationSpace(workload_));

  // Indicates if each of the elements of the array above, was ever updated
  // by a recursive call. Only needed to ensure correctness.
  std::vector<bool> valid_delta(num_spatial_elems, false);

  FillSpatialDeltas(cur, spatial_deltas, valid_delta, 0 /* base_index */);
  
  // Check if each element of spatial_deltas was updated by recursive calls.
  for (auto it : valid_delta)
  {
    ASSERT(it);
  }

  // Restore spatial_id_ to original value.
  spatial_id_ /= num_spatial_elems;

  // Records whether we have accounted for each delta
  // (in each problem dimension) either through
  // 1) Link transfers within current level
  // 2) Multicasted or non-multicasted transfers from previous level

  // Previously, we would first attempt to capture deltas via link
  // transfers. For all other residual deltas, we could compute
  // multicast access factors (1 = unicast). Unfortunately, that
  // led to awkward multicast patterns if deltas that *could* have
  // been multicast were captured via link-transfers.
  // New approach: First calculate multicasts. Then, if using link
  // transfers completely obliterates access to a producer level,
  // use those link transfers only.

  std::vector<problem::PerDataSpace<bool>> unaccounted_delta;
  unaccounted_delta.resize(num_spatial_elems);
  for (uint64_t i = 0; i < num_spatial_elems; i++)
  {
    unaccounted_delta[i].fill(true);
  }

  // auto& cur_state = cur->live_state[spatial_id_];
  //  auto& accesses = nest_state_[cur->level].live_state[spatial_id_].accesses;
  auto& cur_state = nest_state_[cur->level].live_state[spatial_id_];

  problem::PerDataSpace<std::vector<std::uint64_t>>
    accesses_without_link_transfers, accesses_with_link_transfers,
    scatter_factors_without_link_transfers, scatter_factors_with_link_transfers,
    cumulative_hops_without_link_transfers, cumulative_hops_with_link_transfers;

  problem::PerDataSpace<std::vector<std::uint64_t>*>
    accesses, scatter_factors, cumulative_hops;
  
  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    // std::cout<<"intermidiate debugging: "<<cur_state.accesses[pvi].size()<<std::endl;
    // resize to spatial fanouts at this master spatial level
    accesses_without_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    accesses_with_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    
    scatter_factors_without_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    scatter_factors_with_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    
    cumulative_hops_without_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    cumulative_hops_with_link_transfers[pvi].resize(cur_state.accesses[pvi].size());
    
    // spatial fanouts at this master spatial level
    for (unsigned i = 0; i < accesses_without_link_transfers[pvi].size(); i++)
    {
      accesses_without_link_transfers[pvi][i] = 0;
      accesses_with_link_transfers[pvi][i] = 0;

      scatter_factors_without_link_transfers[pvi][i] = 0;
      scatter_factors_with_link_transfers[pvi][i] = 0;
      
      cumulative_hops_without_link_transfers[pvi][i] = 0;
      cumulative_hops_with_link_transfers[pvi][i] = 0;
    }

    // Default: do not use link transfers.
    accesses[pvi] = &accesses_without_link_transfers[pvi];
    scatter_factors[pvi] = &scatter_factors_without_link_transfers[pvi];
    cumulative_hops[pvi] = &cumulative_hops_without_link_transfers[pvi];
  }
  
  ComputeAccurateMulticastedAccesses(cur, spatial_deltas, unaccounted_delta,
                                     accesses_without_link_transfers,
                                     scatter_factors_without_link_transfers,
                                     cumulative_hops_without_link_transfers);

  if (!gEnableLinkTransfers && linked_spatial_level_[level])
  {
    static bool warning_printed = false;
    if (gEnableLinkTransferWarning && !warning_printed)
    {
      std::cerr << "WARNING: disabling link transfer computations. Link transfers "
                << "cause the multicast/scatter signature to change. We need to "
                << "record the impact of each potential multicast/scatter signature. "
                << "FIXME." << std::endl;
      warning_printed = true;
    }
  }

  if (gEnableLinkTransfers && linked_spatial_level_[level])
  {
    // std::cout << "A place of dead code?" << std::endl; 
    // seems like dead since linked_spatial_level_ init same as master_spatial_level_
    // Reset unaccounted delta, and now count with link transfers.
    for (uint64_t i = 0; i < num_spatial_elems; i++)
    {
      unaccounted_delta[i].fill(true);
    }

    problem::PerDataSpace<std::uint64_t> link_transfers;

    ComputeNetworkLinkTransfers(cur, spatial_deltas, unaccounted_delta, link_transfers);

    ComputeAccurateMulticastedAccesses(cur, spatial_deltas, unaccounted_delta,
                                       accesses_with_link_transfers,
                                       scatter_factors_with_link_transfers,
                                       cumulative_hops_with_link_transfers);

    // Compare.
    for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
    {
      // if (problem::Shape::DataSpaceID(pvi) == problem::Shape::DataSpaceID::Weight)
      // {
      //   std::cout << "ACCESSES *WITH* LINK TRANSFERS\n";
      //   for (unsigned i = 0; i < accesses_with_link_transfers[pvi].size(); i++)
      //   {
      //     std::cout << "  " << i << ": " << accesses_with_link_transfers[pvi][i]
      //               << ", scatter: " << scatter_factors_with_link_transfers[pvi][i] << std::endl;
      //   }
      //   std::cout << "ACCESSES *WITHOUT* LINK TRANSFERS\n";
      //   for (unsigned i = 0; i < accesses_without_link_transfers[pvi].size(); i++)
      //   {
      //     std::cout << "  " << i << ": " << accesses_without_link_transfers[pvi][i]
      //               << ", scatter: " << scatter_factors_without_link_transfers[pvi][i] << std::endl;
      //   }
      // }
      
      std::uint64_t total_without = std::accumulate(accesses_without_link_transfers[pvi].begin(),
                                                    accesses_without_link_transfers[pvi].end(),
                                                    static_cast<std::uint64_t>(0));
      std::uint64_t total_with = std::accumulate(accesses_with_link_transfers[pvi].begin(),
                                                 accesses_with_link_transfers[pvi].end(),
                                                 static_cast<std::uint64_t>(0));
      if (total_with < total_without)
      {
        cur_state.link_transfers[pvi] += link_transfers[pvi];
        
        accesses[pvi] = &accesses_with_link_transfers[pvi];
        scatter_factors[pvi] = &scatter_factors_with_link_transfers[pvi];
        cumulative_hops[pvi] = &cumulative_hops_with_link_transfers[pvi];
      }
    }
  }

  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    for (unsigned i = 0; i < cur_state.accesses[pvi].size(); i++)
    {
      cur_state.accesses[pvi][i] += (*accesses[pvi])[i];
        
      // Careful: overwriting scatter factor. The multicast/scatter signature must
      // either be un-initialized, or the accesses must be 0 (special case), or
      // it must match with the updated signature.
      if ((*accesses[pvi])[i] > 0)
      {
        if (cur_state.scatter_factors[pvi][i] == 0)
        {
          cur_state.scatter_factors[pvi][i] = (*scatter_factors[pvi])[i];
          cur_state.cumulative_hops[pvi][i] = (*cumulative_hops[pvi])[i];
        }
        else
        {
          // ****** FIXME ****** track multiple multicast/scatter signatures.
          assert(cur_state.scatter_factors[pvi][i] == (*scatter_factors[pvi])[i]);
        }
      }
    }      
  }

  //  auto& accesses = nest_state_[cur->level].live_state[spatial_id_].accesses;

  // Check that all deltas were accounted for correctly.
  for (uint64_t i = 0; i < num_spatial_elems; i++)
  {
    for (auto& it : unaccounted_delta[i])
    {
      ASSERT(!it);
    }
  }

  // Consistency check.
  for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
  {
    std::uint64_t fanout = 0;
    for (unsigned i = 0; i < cur_state.accesses[pvi].size(); i++)
    {
      fanout += (i+1) * cur_state.scatter_factors[pvi][i];
    }
    
    if (fanout != spatial_fanouts_[cur->level])
    {
      std::cerr << "FATAL: fanout mismatch, computed = " << fanout
                << " actual = " << spatial_fanouts_[cur->level] << std::endl;
      exit(1);
    }
  }  
  
  bool dump = false; // (level >= 4);
  if (dump)
  {
    std::cout << "-------\n";
    std::cout << "SPATIAL LEVEL " << level << std::endl;
    std::cout << "-------\n";

    std::cout << "analysis::LoopState:\n";
    for (int l = level; l < int(nest_state_.size()); l++)
    {
      std::cout << "    Level " << l << ": "
                << nest_state_[l].descriptor.dimension
                << " = " << indices_[l] << std::endl;
    }
    std::cout << "Final Spatial Point Set:\n    ";
    point_set.Print();
  }
}

// Computes deltas needed by the spatial elements in the next level.
// Will update a subset of the elements of spatial_deltas
void NestAnalysis::FillSpatialDeltas(std::vector<analysis::LoopState>::reverse_iterator cur,
                                     std::vector<problem::OperationSpace>& spatial_deltas,
                                     std::vector<bool>& valid_delta,
                                     std::uint64_t base_index,
                                     int depth)
{
  int level = cur->level;

  // base_index determines which element of spatial_deltas
  // is going to be updated at the last recursive call to FillSpatialDeltas.
  // It's value is updated as we recursively call FillSpatialDeltas.
  // Very similar to how spatial_id_ is used to identify the spatial element
  // that we are currently computing the working set for.
  base_index *= cur->descriptor.end;

  if (level == 0)
  {
    // std::uint64_t body_iterations = (cur->descriptor.end - cur->descriptor.start) * num_epochs_;
    // macs_ += body_iterations;
    // to avoid double counting of compute_cycles_
    if (base_index == 0 && spatial_id_ == 0)
    {
      // compute_cycles_ += num_epochs_;
      body_info_.accesses += num_epochs_;
    }

    // No more recursive calls, directly update spatial_deltas.
    for (indices_[level] = cur->descriptor.start;
         indices_[level] < cur->descriptor.end;
         indices_[level] += cur->descriptor.stride)
    {
      std::uint64_t spatial_delta_index = base_index + indices_[level];
      ASSERT(spatial_delta_index < spatial_deltas.size());
      ASSERT(!valid_delta[spatial_delta_index]);

      spatial_deltas[spatial_delta_index] += IndexToOperationPoint_(indices_);
      valid_delta[spatial_delta_index] = true;
    }
  }
  else // level > 0
  {
    auto next = cur + 1;
    int dim = int(cur->descriptor.dimension);
    int scale = per_level_dim_scales_[level][dim];

    // Save state.
    auto orig_spatial_id = spatial_id_;
    auto saved_transform = cur_transform_[dim];

    if (loop::IsSpatial(next->descriptor.spacetime_dimension))
    {
      // Next-inner loop level is spatial. Note that we do not use the
      // gExtrapolateUniformSpatial optimization here. To do that, we need to
      // extrapolate the entire *vector* of spatial_deltas returned by the
      // recursive FillSpatialDeltas() call. TODO.
      for (indices_[level] = cur->descriptor.start;
           indices_[level] < cur->descriptor.end;
           indices_[level] += cur->descriptor.stride)
      {
        ++cur;

        FillSpatialDeltas(cur, spatial_deltas, valid_delta,
                          base_index + indices_[level], depth+1);

        --cur;
        cur_transform_[dim] += scale;
      }
    }
    else // Next-inner loop level is temporal.
    {
      unsigned num_iterations = 1 +
        ((cur->descriptor.end - 1 - cur->descriptor.start) /
         cur->descriptor.stride);

      unsigned iterations_run = 0;
      indices_[level] = cur->descriptor.start;

      unsigned iterations_to_run = gExtrapolateUniformSpatial ? 3 : num_iterations;

      // Run iterations #0, #1, ... #iterations_to_run-1
      for (indices_[level] = cur->descriptor.start;
           indices_[level] < cur->descriptor.end && iterations_run < iterations_to_run;
           indices_[level] += cur->descriptor.stride, iterations_run++)
      {
        ++cur;

        std::uint64_t spatial_delta_index = base_index + indices_[level];
        ASSERT(spatial_delta_index < spatial_deltas.size());
        ASSERT(!valid_delta[spatial_delta_index]);

        spatial_id_ = orig_spatial_id + spatial_delta_index;
        spatial_deltas[spatial_delta_index] = ComputeDeltas(cur);
        valid_delta[spatial_delta_index] = true;

        --cur;
        cur_transform_[dim] += scale;
      }

      // Extrapolate all other iterations.
      if (iterations_run < num_iterations)
      {
        // Determine translation vector from #iterations_to_run-2 to #iterations_to_run-1.
        std::vector<Point> translation_vectors;

        auto& opspace_lastrun = spatial_deltas[base_index + indices_[level] - cur->descriptor.stride];
        auto& opspace_secondlastrun = spatial_deltas[base_index + indices_[level] - 2*cur->descriptor.stride];

        for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
        {
          translation_vectors.push_back(
            opspace_secondlastrun.GetDataSpace(pv).GetTranslation(opspace_lastrun.GetDataSpace(pv)));
        }

        // Iterations #num_iterations_to_run through #last.
        problem::OperationSpace* prev_temporal_delta = &opspace_lastrun;
        for (;
             indices_[level] < cur->descriptor.end;
             indices_[level] += cur->descriptor.stride, iterations_run++)
        {
          std::uint64_t spatial_delta_index = base_index + indices_[level];
          ASSERT(spatial_delta_index < spatial_deltas.size());
          ASSERT(!valid_delta[spatial_delta_index]);

          spatial_id_ = orig_spatial_id + spatial_delta_index;

          auto& temporal_delta = spatial_deltas[spatial_delta_index];
          for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
          {
            temporal_delta.GetDataSpace(pv) = prev_temporal_delta->GetDataSpace(pv);
            temporal_delta.GetDataSpace(pv).Translate(translation_vectors.at(pv));
          }
          valid_delta[spatial_delta_index] = true;

          prev_temporal_delta = &temporal_delta;
        } // extrapolated iterations
      } // iterations_run < num_iterations
    } // next inner loop is temporal

    // Restore state.
    cur_transform_[dim] = saved_transform;
    spatial_id_ = orig_spatial_id;

  } // level > 0  
}

// Exhaustively compare all pairs of deltas and infer multicast opportunities.
void NestAnalysis::ComputeAccurateMulticastedAccesses(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::vector<problem::OperationSpace>& spatial_deltas,
    std::vector<problem::PerDataSpace<bool>>& unaccounted_delta,
    problem::PerDataSpace<std::vector<std::uint64_t>>& accesses,
    problem::PerDataSpace<std::vector<std::uint64_t>>& scatter_factors,
    problem::PerDataSpace<std::vector<std::uint64_t>>& cumulative_hops)
{
  std::uint64_t num_deltas = spatial_deltas.size();

  // For each data type, records the number of unaccounted deltas
  // that the current delta matches with. This will be used
  // to infer the multicast factor for a specific delta.
  // reused across loop iterations to avoid initialization overheads.
  problem::PerDataSpace<uint64_t> num_matches;

  // For each datatype, records a ve
  
  // std::cout << "-----------------------------\n";
  // std::cout << "       COMPUTE MULTICAST     \n";
  // std::cout << "-----------------------------\n";
  // std::cout << "Epochs = " << num_epochs_ << std::endl;
  // std::cout << "Num deltas = " << num_deltas << std::endl;

  // for (std::uint64_t i = 0; i < num_deltas; i++)
  // {
  //   auto pv = problem::Shape::DataSpaceID::Weight;
  //   if (unaccounted_delta[i][int(pv)])
  //     std::cout << "UNACCOUNTED: ";
  //   else
  //     std::cout << "  ACCOUNTED: ";
  //   spatial_deltas[i].Print(pv);
  // }
  
  auto h_size = horizontal_sizes_[cur->level];
  auto v_size = vertical_sizes_[cur->level];

  for (std::uint64_t i = 0; i < num_deltas; i++)
  {
    num_matches.fill(0);
    
    problem::PerDataSpace<std::vector<std::uint64_t>> match_set;

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (!unaccounted_delta[i][pv])
      {
        // this delta was already accounted for,
        // skip the comparisons.
        continue;
      }

      unaccounted_delta[i][pv] = false;
      num_matches[pv] = 1;  // we match with ourselves.
      match_set[pv].push_back(i);

      for (std::uint64_t j = i + 1; j < num_deltas; j++)
      {
        if (unaccounted_delta[j][pv])
        {
          if (spatial_deltas[i].CheckEquality(spatial_deltas[j], pv))
          {
            // We have a match, record it
            unaccounted_delta[j][pv] = false;
            num_matches[pv]++;
            match_set[pv].push_back(j);
          }
        }
      }
    }

    // update the number of accesses at different multicast factors.
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (num_matches[pv] > 0)
      {
        accesses[pv][num_matches[pv] - 1] += (spatial_deltas[i].GetSize(pv) * num_epochs_);
        scatter_factors[pv][num_matches[pv] - 1]++;

        // Compute the average number of hops from the edge of the array
        // (at this level) to the nodes in the match set.
        // Assume injection point is at center of V-axis. Routing algorithm is
        // to go along H maximally, then drop vertical paths.

        ASSERT(num_matches[pv] == match_set[pv].size());
        
        double hops = 0;
        
        std::uint64_t h_max = 0;
        for (auto& linear_id : match_set[pv])
        {
          std::uint64_t h_id = linear_id % h_size;
          h_max = std::max(h_max, h_id);
        }
        hops += double(h_max);
        
        double v_center = double(v_size-1) / 2;
        for (auto& linear_id : match_set[pv])
        {
          std::uint64_t v_id = linear_id / h_size;
          hops += std::abs(double(v_id) - v_center);
        }

        // Accumulate this into the running hop count. We'll finally divide this
        // by the scatter factor to get average hop count.
        cumulative_hops[pv][num_matches[pv] - 1] += hops;
      }
    }
  }
}

// Compares two deltas, and if they are equal,
// records the opportunity for inter-PE link transfers.
void CompareSpatioTemporalDeltas(
    const std::vector<problem::OperationSpace>& cur_spatial_deltas,
    const std::vector<problem::OperationSpace>& prev_spatial_deltas,
    const std::uint64_t cur_spatial_index,
    const std::uint64_t prev_spatial_index,
    std::vector<problem::PerDataSpace<bool>>& inter_elem_reuse)
{
  auto& cur_delta = cur_spatial_deltas[cur_spatial_index];
  auto& prev_delta = prev_spatial_deltas[prev_spatial_index];

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!cur_delta.IsEmpty(pv))
    {
      if (cur_delta.CheckEquality(prev_delta, pv))
      {
        // ASSERT(!inter_elem_reuse[cur_spatial_index][pv]);
        inter_elem_reuse[cur_spatial_index][pv] = true;
      }
    }
  }
}

void NestAnalysis::ComputeNetworkLinkTransfers(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::vector<problem::OperationSpace>& cur_spatial_deltas,
    std::vector<problem::PerDataSpace<bool>>&
    unaccounted_delta,
    problem::PerDataSpace<std::uint64_t>& link_transfers)
{
  // std::cout << "-----------------------------\n";
  // std::cout << "         LINK TRANSFERS      \n";
  // std::cout << "-----------------------------\n";
  
  // std::cout << "CUR BEFORE:" << std::endl;
  // for (std::uint64_t i = 0; i < cur_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::Shape::DataSpaceID::Weight;
  //   if (unaccounted_delta[i][int(pv)])
  //     std::cout << "  UNACCOUNTED: ";
  //   else
  //     std::cout << "    ACCOUNTED: ";
  //   cur_spatial_deltas[i].Print(pv);
  // }
  
  auto h_size = horizontal_sizes_[cur->level];
  auto v_size = vertical_sizes_[cur->level];

  // Imagine origin (0,0) at the top-left corner of a 2D spatial array.
  // Horizontal ids grow from left to right.
  // Vertical ids grow from top to bottom.
  auto GetLinearIndex = [&h_size, &v_size](std::uint64_t h_id, std::uint64_t v_id)
    {
      ASSERT(h_id < h_size && v_id < v_size);
      std::uint64_t linearIndex = v_id * h_size + h_id;  // row major layout
      return linearIndex;
    };

  auto& cur_state = cur->live_state[spatial_id_];
  auto& prev_spatial_deltas = cur_state.prev_point_sets[0];
  ASSERT(cur_spatial_deltas.size() == prev_spatial_deltas.size());
  int num_spatial_elems = spatial_fanouts_[cur->level];

  // std::cout << "PREV:" << std::endl;
  // for (std::uint64_t i = 0; i < prev_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::Shape::DataSpaceID::Weight;
  //   std::cout << "  "; prev_spatial_deltas[i].Print(pv);
  // }
  
  // for each spatial elements, this array records if the data
  // needed by the element can be obtained from any of the neighboring elements.
  std::vector<problem::PerDataSpace<bool>> inter_elem_reuse;
  inter_elem_reuse.resize(num_spatial_elems);
  for (int i = 0; i < num_spatial_elems; i++)
  {
    inter_elem_reuse[i].fill(false);
  }

  // FIXME The loops below can be codified in some way to avoid redundant LOC.

  // Test for a few hard-coded transfer patterns in horizontal and vertical
  // dimensions.

  // downward vertical transfers in each column
  for (std::uint64_t h_id = 0; h_id < h_size; h_id++)
  {
    for (std::uint64_t v_id = 1; v_id < v_size; v_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id, v_id - 1);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // upward vertical transfers in each column
  for (std::uint64_t h_id = 0; h_id < h_size; h_id++)
  {
    for (std::uint64_t v_id = 0; v_id < v_size - 1; v_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id, v_id + 1);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // horizontal transfers in each row from left to right
  for (std::uint64_t v_id = 0; v_id < v_size; v_id++)
  {
    for (std::uint64_t h_id = 1; h_id < h_size; h_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id - 1, v_id);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // horizontal transfers in each row from right to left
  for (std::uint64_t v_id = 0; v_id < v_size; v_id++)
  {
    for (std::uint64_t h_id = 0; h_id < h_size - 1; h_id++)
    {
      auto cur_spatial_index = GetLinearIndex(h_id, v_id);
      auto prev_spatial_index = GetLinearIndex(h_id + 1, v_id);
      CompareSpatioTemporalDeltas(cur_spatial_deltas, prev_spatial_deltas,
                                  cur_spatial_index, prev_spatial_index,
                                  inter_elem_reuse);
    }
  }

  // Compute the total number of accesses that can be bypassed
  // by using link transfers
  for (int i = 0; i < num_spatial_elems; i++)
  {
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      if (inter_elem_reuse[i][pv])
      {
        link_transfers[pv] += (cur_spatial_deltas[i].GetSize(pv) * num_epochs_);
        ASSERT(unaccounted_delta[i][pv]);
        unaccounted_delta[i][pv] = false;
      }
    }
  }

  // Time-shift the data in prev_point_sets array
  for (std::uint64_t i = 1; i < analysis::ElementState::MAX_TIME_LAPSE; i++)
  {
    for (int j = 0; j < num_spatial_elems; j++)
    {
      cur_state.prev_point_sets[i - 1][j] = cur_state.prev_point_sets[i][j];
    }
  }

  for (int j = 0; j < num_spatial_elems; j++)
  {
    cur_state.prev_point_sets[analysis::ElementState::MAX_TIME_LAPSE - 1][j] =
        cur_spatial_deltas[j];
  }

  // std::cout << "AFTER:" << std::endl;
  // for (std::uint64_t i = 0; i < cur_spatial_deltas.size(); i++)
  // {
  //   auto pv = problem::Shape::DataSpaceID::Weight;
  //   if (unaccounted_delta[i][int(pv)])
  //     std::cout << "  UNACCOUNTED: ";
  //   else
  //     std::cout << "    ACCOUNTED: ";
  //   cur_spatial_deltas[i].Print(pv);
  // }
}

// computes the number of spatial elements at each level
// and identifies master spatial levels.
void NestAnalysis::InitNumSpatialElems()
{
  num_spatial_elems_.resize(nest_state_.size());
  master_spatial_level_.resize(nest_state_.size());

  int cur_index = nest_state_.size() - 1;
  // cumulative product of spatial tiling factors.
  std::uint64_t product = 1;
  bool prev_loop_was_spatial = false;
  // from outer loop to inner loop
  for (auto loop = nest_state_.rbegin(); loop != nest_state_.rend(); loop++)
  {
    ASSERT(cur_index >= 0);

    num_spatial_elems_[cur_index] = product;
    if (loop::IsSpatial(loop->descriptor.spacetime_dimension))
    {
      // master spatial level: most outer loop for consequtive spatial levels
      master_spatial_level_[cur_index] = !prev_loop_was_spatial;
      product *= loop->descriptor.end;
      prev_loop_was_spatial = true;
    }
    else
    {
      master_spatial_level_[cur_index] = false;
      prev_loop_was_spatial = false;
    }

    cur_index--;
  }

  linked_spatial_level_.resize(nest_state_.size(), false);
  for (std::uint64_t cur_level = 0; cur_level < nest_state_.size(); cur_level++)
  {
    if (master_spatial_level_[cur_level])
    {
      linked_spatial_level_[cur_level] = true;
    }
  }

  if(verbose_){
    std::cout << "Number of spatial elements at each level from outer loop to inner loop" << std::endl;
    for (int i = num_spatial_elems_.size() - 1; i >= 0; i--) // nest_state_.size()
    {
      std::cout << num_spatial_elems_[i];
      if (master_spatial_level_[i]) std::cout << "(master)";
      if (linked_spatial_level_[i]) std::cout << "(linked)";
      std::cout << ", ";
    }
    std::cout << std::endl;
  }
}

void NestAnalysis::InitStorageBoundaries()
{
  storage_boundary_level_.resize(nest_state_.size(), false);
  for (auto& i : storage_tiling_boundaries_)
  {
    ASSERT(i < storage_boundary_level_.size());
    storage_boundary_level_[i] = true;
  }
}

void NestAnalysis::InitSpatialFanouts()
{
  spatial_fanouts_.resize(nest_state_.size(), 1);
  horizontal_sizes_.resize(nest_state_.size(), 1);
  vertical_sizes_.resize(nest_state_.size(), 1);
  for (int cur_level = nest_state_.size() - 1; cur_level >= 0; cur_level--)
  {
    if (!loop::IsSpatial(nest_state_[cur_level].descriptor.spacetime_dimension)) // temporal levels
    {
      spatial_fanouts_[cur_level] = 1;
    }
    else if (!master_spatial_level_[cur_level]) // non-master spatial level 
    {
      spatial_fanouts_[cur_level] = 0;
    }
    else // master spatial level
    {
      int next_temporal_level = cur_level;
      int scale_factor = 1;
      while (loop::IsSpatial(nest_state_[next_temporal_level].descriptor.spacetime_dimension))
      {
        if (loop::IsSpatialX(nest_state_[next_temporal_level].descriptor.spacetime_dimension))
        {
          horizontal_sizes_[cur_level] *=
              nest_state_[next_temporal_level].descriptor.end;
        }
        else
        {
          vertical_sizes_[cur_level] *=
              nest_state_[next_temporal_level].descriptor.end;
        }

        if (next_temporal_level > 0)
        {
          next_temporal_level--; // go to inner loop
        }
        else
        {
          scale_factor = nest_state_[0].descriptor.end;
          break;
        }
      } // loop exit when we exit consequtive spatial loops

      spatial_fanouts_[cur_level] =
          num_spatial_elems_[next_temporal_level] / num_spatial_elems_[cur_level];
      spatial_fanouts_[cur_level] *= scale_factor;

      ASSERT(spatial_fanouts_[cur_level] ==
             horizontal_sizes_[cur_level] * vertical_sizes_[cur_level]);
    }
  }

  if(verbose_)
  {
    std::cout << "Spatial fanouts at each level from outer to inner loop" << std::endl;
    for (int i = num_spatial_elems_.size() - 1; i >= 0; i--)
    {
      std::cout << spatial_fanouts_[i];
      std::cout << ", ";
    }
    std::cout << std::endl;
  }
}

// Related to Memoization structures
void NestAnalysis::InitPerLevelDimScales()
{
  // cur_transform_ is a OperationPoint
  // mold_low_, mold_high_ is a vector of OperationPoint with size equal to num of loop levels
  for (unsigned dim = 0; dim < problem::GetShape()->NumDimensions; dim++)
  {
    cur_transform_[dim] = 0;
  }

  std::uint64_t num_levels = nest_state_.size();

  per_level_dim_scales_.resize(num_levels);
  mold_low_.resize(num_levels);
  mold_high_.resize(num_levels);

  // running scale maintained for each dimension.
  problem::PerProblemDimension<std::uint64_t> cur_scale;
  cur_scale.fill(1);

  for (std::uint64_t level = 0; level < num_levels; level++)
  {
    auto desc = nest_state_[level].descriptor;
    int dim = int(desc.dimension);

    for (std::uint64_t dim = 0; dim < problem::GetShape()->NumDimensions; dim++)
    {
      per_level_dim_scales_[level][dim] = cur_scale[dim];
    }

    cur_scale[dim] *= (desc.end - desc.start);  // FIXME: assuming stride = 1

    for (std::uint64_t dim = 0; dim < problem::GetShape()->NumDimensions; dim++)
    {
      mold_low_[level][dim] = desc.start;
      mold_high_[level][dim] = cur_scale[dim] - 1; // FIXME: this is wrong.
    }
  }

  if(verbose_)
  {
    std::cout << "per level dimension scale from inner to outer loop" << std::endl;
    for (std::uint64_t level = 0; level < num_levels; level++)
    {
      std::cout <<"level: "<< level <<std::endl;
      for (std::uint64_t dim = 0; dim < problem::GetShape()->NumDimensions; dim++){
        std::cout << problem::GetShape()->DimensionIDToName.at(dim) << ": " << per_level_dim_scales_[level][dim]<< "  ";
      }
      std::cout << std::endl;

      for (std::uint64_t dim = 0; dim < problem::GetShape()->NumDimensions; dim++){
        std::cout << problem::GetShape()->DimensionIDToName.at(dim) << ": " << mold_high_[level][dim]<< "  ";
      }    
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// Transform an index to a problem point.

// arm: This routine is called a lot of times (no. of MACs in CONV layer),
// But, it is totally unoptimized and does a lot of redundant computation.
// There is a not-so-complicated way to optimize this
// by exploiting the global loop nest information.
// instead of making naive local decisions.
problem::OperationPoint NestAnalysis::IndexToOperationPoint_(
  const std::vector<int>& indices) const
{
  problem::OperationPoint point;
  for (unsigned dim = 0; dim < problem::GetShape()->NumDimensions; dim++)
  {
    point[dim] = 0;
  }

  for (unsigned int level = 0; level < indices.size(); level++)
  {
    auto desc = nest_state_[level].descriptor;
    int dim = int(desc.dimension);
    point[dim] += (per_level_dim_scales_[level][dim] * indices[level]);
  }

  return point;
}

// A heuristic way to infer multicast opportunities.
// Will correctly identify multicasts when data type
// indices don't depend on multiple problem indices.
// (Ex. Weights and Outputs)
// When data type indices depend on multiple problem indices
// (Ex. Inputs), we break the assumption that multicast
// inference can be done at a per-level basis.
// Not Used in codebase
void NestAnalysis::ComputeApproxMulticastedAccesses(
    std::vector<analysis::LoopState>::reverse_iterator cur,
    const std::vector<problem::OperationSpace>& spatial_deltas)
{
  // Find number of spatial levels that correspond to this master spatial level.
  int master_level = cur->level;
  uint64_t num_spatial_levels;
  {
    int next_temporal_level = master_level;
    while (loop::IsSpatial(nest_state_[next_temporal_level].descriptor.spacetime_dimension) &&
           next_temporal_level > 0)
    {
      next_temporal_level--;
    }
    if (next_temporal_level == 0 && loop::IsSpatial(nest_state_[0].descriptor.spacetime_dimension))
    {
      next_temporal_level--;
    }
    num_spatial_levels = cur->level - next_temporal_level;
  }

  // for each level, stores if the tiling at that level results in multicasting
  // for any of the problem variables.
  problem::PerDataSpace<std::vector<bool>>
      is_multicast_level;  // per-pv, per-level
  for (auto& it : is_multicast_level)
  {
    it.resize(num_spatial_levels, false);
  }

  std::vector<uint64_t> max_vals(num_spatial_levels);
  std::vector<uint64_t> cur_vals(num_spatial_levels, 0);
  for (uint64_t i = 0; i < num_spatial_levels; i++)
  {
    max_vals[i] = nest_state_[master_level - i].descriptor.end;
  }

  auto GetSpatialIndex = [&max_vals, &cur_vals]() {
    uint64_t final_index = 0;
    uint64_t scale = 1;
    for (int i = max_vals.size() - 1; i >= 0; i--)
    {
      final_index += scale * cur_vals[i];
      scale *= max_vals[i];
    }
    return final_index;
  };

  for (uint64_t level = 0; level < num_spatial_levels; level++)
  {
    std::vector<uint64_t> indices_to_compare;
    for (uint64_t j = 0; j < max_vals[level]; j++)
    {
      cur_vals[level] = j;
      indices_to_compare.push_back(GetSpatialIndex());
    }
    cur_vals[level] = 0;  // reset

    problem::PerDataSpace<bool> is_multicast;
    is_multicast.fill(true);
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      for (uint64_t i = 1; i < indices_to_compare.size(); i++)
      {
        auto lhs_index = indices_to_compare[i];
        auto rhs_index = indices_to_compare[i - 1];
        if (!spatial_deltas[lhs_index]
                 .CheckEquality(spatial_deltas[rhs_index], pv))
        {
          is_multicast[pv] = false;
          break;
        }
      }
    }

    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      is_multicast_level[pv][level] = is_multicast[pv];
    }
  }

  problem::PerDataSpace<std::size_t> summed_deltas;
  summed_deltas.fill(0);
  for (uint64_t i = 0; i < spatial_deltas.size(); i++)
  {
    auto delta_sizes = spatial_deltas[i].GetSizes();
    for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
    {
      summed_deltas[pv] += delta_sizes[pv];
    }
  }

  problem::PerDataSpace<std::size_t> multicast_factors;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    uint64_t product_of_multicast_levels = 1;
    for (uint64_t level = 0; level < num_spatial_levels; level++)
    {
      if (is_multicast_level[pv][level])
      {
        product_of_multicast_levels *= max_vals[level];
      }
    }
    multicast_factors[pv] = product_of_multicast_levels;
  }

  // compute and update the number of accesses at various multicast factors.
  auto& accesses = nest_state_[master_level].live_state[spatial_id_].accesses;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    ASSERT(accesses[pv].size() == spatial_deltas.size());
    ASSERT(summed_deltas[pv] % multicast_factors[pv] == 0);
    accesses[pv][multicast_factors[pv] - 1] +=
        (summed_deltas[pv] / multicast_factors[pv] * num_epochs_);
  }
}

} // namespace analysis
