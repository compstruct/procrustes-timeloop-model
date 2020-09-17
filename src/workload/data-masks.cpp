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

#include "data-masks.hpp"
#include "workload.hpp"
// #include "operation-space.hpp"

bool gDeterministicMask = (getenv("TIMELOOP_DETERMINISTIC_MASK") != NULL);


namespace problem
{

// ======================================== //
//         Mask Loading/ Generation         //
// ======================================== //

DataMasks::DataMasks(Workload* wc, bool synthetic) : synthetic_(synthetic)
{
  assert(wc != NULL);
  workload_ = wc;
}

void DataMasks::LoadActualMasks(std::string model_name, std::string layer_name)
{
  std::string weight_mask_file_name;
  std::string weight_mask_dir;

  const char* weight_mask_dir_env = std::getenv("TIMELOOP_WEIGHT_MASK_DIR");
  if (weight_mask_dir_env){
    weight_mask_dir = std::string(weight_mask_dir_env) + "/" + model_name + "/";
  }
  else
    weight_mask_dir = "/home/dingqing/backprop/vgg/";
  
  if(!layer_name.empty())
    weight_mask_file_name = weight_mask_dir + model_name + "_" + layer_name + ".npy";
  else
  {
    std::cerr << "Reading Weight Mask failed! "<<std::endl;
    assert(false);
  }
  
  weight_mask_ = cnpy::npy_load(weight_mask_file_name);

  // TODO: we should try and catch if file loading fails
  mask_array_ = weight_mask_.data<bool>(); // this has to be a member, not local variable
  // otherwise all related member will be deleted

  auto weight_order = GetWeightOrder();
  if(weight_order == 3) // depthwise
  {
    // processing multiple samples
    if(weight_mask_.shape.size()==4)
    {
      total_samples_ = weight_mask_.shape[0]; // the last dim is sample size
    }
    else if (weight_mask_.shape.size()==3)
    {
      total_samples_ = 1;
    }
    else
    {
      std::cout << "Loaded weight mask npy file has the wrong shape: " << weight_mask_.shape.size() << std::endl;
      assert(false);
    }

    // reshape 2D weight mask to 4D for consistency if FC
    unsigned offset = (total_samples_ == 1) ? 0 : 1;

    data_mask_shape_.resize(weight_order);
    for (unsigned i = 0; i < weight_order; i++)
    {
      data_mask_shape_[i] = weight_mask_.shape[i+offset];
    }
  }
  else if (weight_order == 4) // normal conv or FC
  {
    // processing multiple samples
    if(weight_mask_.shape.size()==3 || weight_mask_.shape.size()==5)
    {
      total_samples_ = weight_mask_.shape[0]; // the last dim is sample size
    }
    else
    {
      total_samples_ = 1;
    }

    // reshape 2D weight mask to 4D for consistency
    unsigned offset = (total_samples_ == 1) ? 0 : 1;

    data_mask_shape_.resize(weight_order);

    if(weight_mask_.shape.size()==2+offset)
    {
      data_mask_shape_[0] = 1; // R
      data_mask_shape_[1] = 1; // S
      data_mask_shape_[2] = weight_mask_.shape[0+offset]; // C
      data_mask_shape_[3] = weight_mask_.shape[1+offset]; // K
    } 
    else if (weight_mask_.shape.size()==4+offset)
    {
      for(unsigned i=0; i<data_mask_shape_.size();i++)
        data_mask_shape_[i] = weight_mask_.shape[i+offset];
    }
    else
      assert(false); // we only allow 2D and 4D weight mask
  }

  CheckWeightBounds();

#if 1
  std::cout << "First 30 mask loaded: "<<std::endl;
  for (unsigned i = 0; i < 30; i++)
    std::cout << mask_array_[i] << " ";
  std::cout <<std::endl;
#endif
}


void DataMasks::GenerateSyntheticMasks(double target_sparsity, std::string save)
{
  std::random_device rd;
  auto seed = gDeterministicMask ? 0 : rd();
  std::mt19937 gen(seed);
  std::bernoulli_distribution d(target_sparsity);

  auto weight_order = GetWeightOrder();
  data_mask_shape_.resize(weight_order);
  if(IsDWWU()) // this might be simplified if the projection in problem file is alligned (dingqing FIXME: code simplification)
  {
    data_mask_shape_[0] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("P")));
    data_mask_shape_[1] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("Q")));
    data_mask_shape_[2] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("C")));
    data_mask_shape_[3] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("N")));
  }
  else
  {
    data_mask_shape_[0] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("R")));
    data_mask_shape_[1] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("S")));
    data_mask_shape_[2] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("C")));
    if(weight_order == 4)
      data_mask_shape_[3] = static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("K")));
  }

  total_samples_ = 1;
  unsigned target_size = std::accumulate(data_mask_shape_.begin(), data_mask_shape_.end(), static_cast<unsigned>(1), std::multiplies<>());
  std::cout <<"test bernoulli generation: "<<std::endl;
  std::cout <<"target_sparsity: "<<target_sparsity<<std::endl;
  mask_array_ = new bool[target_size];
  for(unsigned i=0;i<target_size; i++)
  {
    mask_array_[i] = d(gen);
  }
  if (!save.empty())
    cnpy::npy_save(save + ".npy", mask_array_, data_mask_shape_, "w");

#if 1
  std::cout << "First 30 mask during generation: "<<std::endl;
  for (unsigned i = 0; i < 30; i++)
    std::cout << mask_array_[i] << " ";
  std::cout <<std::endl;
#endif
}

void DataMasks::CheckWeightBounds()
{  
  // dingding optional FIXME: for better generality
  // check match with shape dimension
  assert(data_mask_shape_[0] == static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("R"))));
  assert(data_mask_shape_[1] == static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("S"))));
  assert(data_mask_shape_[2] == static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("C"))));
  if(GetWeightOrder()==4)
    assert(data_mask_shape_[3] == static_cast<size_t>(workload_->GetBound(GetShape()->DimensionNameToID.at("K"))));
}

unsigned DataMasks::GetWeightOrder()
{
  if(IsDWWU())
    return workload_->GetShape()->DataSpaceOrder.at(GetShape()->DataSpaceNameToID.at("Inputs"));
  else
    return workload_->GetShape()->DataSpaceOrder.at(GetShape()->DataSpaceNameToID.at("Weights"));
}

bool DataMasks::IsDWWU()
{
  return workload_->GetShape()->name == "Weight-Update-Depthwise";
}


}  // namespace problem
