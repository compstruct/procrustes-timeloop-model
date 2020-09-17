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

// #include <map>
#include <vector>
// #include <list>
#include <random>
#include "cnpy.h"

#include "workload.hpp"

namespace problem
{

// static std::string layerType[4] = { "normalConv",
//                                     "depthwiseConv",
//                                     "bottleneckConv",
//                                     "FC"};

class DataMasks
{
 private:
  bool synthetic_;
  Workload* workload_ = nullptr;
  std::vector<size_t> data_mask_shape_;
  // std::string type_;
  cnpy::NpyArray weight_mask_; // this must be a member, otherwise the data field will be released
  bool* mask_array_=nullptr;
  unsigned total_samples_;

  // auxiliary
  void CheckWeightBounds();
  unsigned GetWeightOrder();
  bool IsDWWU();
  // void DetermineLayerType(); // not clear if useful

 public: 
  DataMasks()=delete;
  DataMasks(Workload* wc, bool synthetic);
  // DataMasks(std::string model_name, std::string layer_name, problem::Workload* wc);
  // DataMasks(double target_sparsity, problem::Workload* wc, std::string save=std::string());
  ~DataMasks()
  {
    if(synthetic_)
      delete[] mask_array_;
  }

  // getter
  std::vector<size_t> getDataMaskShape(){ return data_mask_shape_;}
  std::vector<size_t> getDataMaskShape() const { return data_mask_shape_;} 
  // std::string getLayerType() { return type_; }
  // std::string getLayerType() { return type_; }
  bool* getMaskPtr() { return mask_array_; }
  bool* getMaskPtr() const { return mask_array_; }
  unsigned getTotalSamples() { return total_samples_; }
  unsigned getTotalSamples() const { return total_samples_; }

  // high level loading/generation api
  void LoadActualMasks(std::string model_name, std::string layer_name);
  void GenerateSyntheticMasks(double target_sparsity, std::string save=std::string());
};

} // namespace problem