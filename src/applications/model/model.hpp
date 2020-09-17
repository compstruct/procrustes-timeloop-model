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

#include <fstream>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "util/accelergy_interface.hpp"
#include "util/banner.hpp"
#include "mapping/parser.hpp"
#include "compound-config/compound-config.hpp"
// #include "workload/data-masks.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

class Application
{
 protected:
  // Critical state.
  problem::Workload workload_;
  model::Engine::Specs arch_specs_;
  std::string model_name_;
  std::string layer_name_;
  
  problem::DataMasks* tracked_weight_masks_;
  
  // weight mask info
  bool is_mask_synthetic_;
  double target_sparsity_;
  std::string synthetic_mask_path_;

  // The mapping has to be a dynamic object because we cannot
  // instantiate it before the problem shape has been parsed. UGH.
  Mapping* mapping_;

  // Application flags.
  bool verbose_ = false; // their verbose flag only apply to this file only (has nothing to do with my verbose in evaluation)
  bool auto_bypass_on_failure_ = false;

 private:

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0)
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(workload_);
    }
  }

 public:

  Application(config::CompoundConfig* config)
  {    
    auto rootNode = config->getRoot();

    // Model application configuration.
    auto_bypass_on_failure_ = false;

    if (rootNode.exists("model"))
    {
      auto model = rootNode.lookup("model");
      model.lookupValue("verbose", verbose_);
      model.lookupValue("auto_bypass_on_failure", auto_bypass_on_failure_);
    }
    if (verbose_)
    {
      for (auto& line: banner)
        std::cout << line << std::endl;
      std::cout << std::endl;
    }

    // Problem configuration.
    auto problem = rootNode.lookup("problem");
    problem::ParseWorkload(problem, workload_);
    if (verbose_)
      std::cout << "Problem configuration complete." << std::endl;

    // Architecture configuration.
    config::CompoundConfigNode arch;
    if (rootNode.exists("arch"))
    {
      arch = rootNode.lookup("arch");
    }
    else if (rootNode.exists("architecture"))
    {
      arch = rootNode.lookup("architecture");
    }
    arch_specs_ = model::Engine::ParseSpecs(arch);

    if (rootNode.exists("ERT"))
    {
      auto ert = rootNode.lookup("ERT");
      if (verbose_)
        std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
    }
    else
    {
#ifdef USE_ACCELERGY
      // Call accelergy ERT with all input files
      if (arch.exists("subtree") || arch.exists("local"))
      {
        accelergy::invokeAccelergy(config->inFiles);
        auto ertConfig = new config::CompoundConfig("ERT.yaml");
        auto ert = ertConfig->getRoot().lookup("ERT");
        if (verbose_)
          std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
        arch_specs_.topology.ParseAccelergyERT(ert);
      }
#endif
    }

    if (verbose_)
      std::cout << "Architecture configuration complete." << std::endl;

    // Mapping configuration: expressed as a mapspace or mapping.
    auto mapping = rootNode.lookup("mapping");
    mapping_ = new Mapping(mapping::ParseAndConstruct(mapping, arch_specs_, workload_));
    if (verbose_)
      std::cout << "Mapping construction complete." << std::endl;

    auto mapper = rootNode.lookup("mapper"); // pass layer info
    // Weight Mask
    model_name_ = "vgg";
    layer_name_ = "Conv5_2"; // specifies weight mask name: eg Conv5_3
    mapper.lookupValue("model-name", model_name_);
    mapper.lookupValue("layer-name", layer_name_);

    is_mask_synthetic_ = mapper.exists("mask-synthetic");
    std::cout << "mask synthetic? "<<is_mask_synthetic_<<std::endl;
    target_sparsity_ = 0; // default
    synthetic_mask_path_ = std::string(); // default empty string
    // TODO: make this config hierarchy (list maybe)
    if(is_mask_synthetic_)
    {
      auto mask_parser = mapper.lookup("mask-synthetic");
      // target_sparsity_ = 0.5; // default if synthetic
      mask_parser.lookupValue("target-sparsity", target_sparsity_);
      // TODO: now we put in current dir, later lets put it somewhere close to actual mask
      synthetic_mask_path_ = std::string(); // empty string
      mask_parser.lookupValue("synthetic-mask-path", synthetic_mask_path_);
      std::cout << "target sparsity: "<<target_sparsity_<<std::endl;
      std::cout << "synthetic-mask-path: "<<synthetic_mask_path_<<std::endl;
    }

    if (verbose_)
    {
      std::cout << "model name: "<<model_name_<<std::endl;
      std::cout << "layer name: "<<layer_name_<<std::endl;
    }
  }

  // This class does not support being copied
  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;

  ~Application()
  {
    if (mapping_)
      delete mapping_;
  }

  // Run the evaluation.
  void Run()
  {
    // Output file names.
    const std::string out_prefix = "timeloop-model.";
    const std::string stats_file_name = out_prefix + "stats.txt";
    const std::string xml_file_name = out_prefix + "map+stats.xml";
    const std::string map_txt_file_name = out_prefix + "map.txt";

    model::Engine engine;
    engine.Spec(arch_specs_);

    auto level_names = arch_specs_.topology.LevelNames();

    auto& mapping = *mapping_;

    tracked_weight_masks_ = new problem::DataMasks(&workload_, is_mask_synthetic_); // randomly generated masks

    // parsing is_mask_synthetic_ missing
    if(!is_mask_synthetic_)
      tracked_weight_masks_->LoadActualMasks(model_name_, layer_name_);
    else
      tracked_weight_masks_->GenerateSyntheticMasks(target_sparsity_, synthetic_mask_path_);
    
    // Now need preEvaluate to check available buffer budget
    auto pre_eval_status = engine.PreEvaluationCheck(mapping, workload_, tracked_weight_masks_, true); 
    for (unsigned level = 0; level < pre_eval_status.size(); level++)
      if (!pre_eval_status[level].success)
      {
        if(verbose_)
          std::cerr << "WARNING: couldn't map level " << level_names.at(level) << ": "
                    << pre_eval_status[level].fail_reason 
                    << std::endl;
        // Optional feature: if the given mapping does not fit in the available
        // hardware resources, automatically bypass storage level(s) to make it
        // fit. This avoids mapping failures and instead substitutes the given
        // mapping with one that fits but is higher cost and likely sub-optimal.
        // *However*, this only covers capacity failures due to temporal factors,
        // not instance failures due to spatial factors. It also possibly
        // over-corrects since it bypasses *all* data-spaces at a failing level,
        // while it's possible that bypassing a subset of data-spaces may have
        // caused the mapping to fit.
        if(!auto_bypass_on_failure_)
        {
          std::cerr << "PreEvaluation Fails! Skip Evaluation." << std::endl;
          return;
        }
        else // auto_bypass_on_failure_
        {
          if(verbose_)
            std::cerr << "Auto Bypassing" << std::endl;
          for (unsigned pvi = 0; pvi < problem::GetShape()->NumDataSpaces; pvi++)
            // Ugh... mask is offset-by-1 because level 0 is the arithmetic level.
            mapping.datatype_bypass_nest.at(pvi).reset(level-1);
        }
      }

    auto eval_status = engine.Evaluate(mapping, workload_, tracked_weight_masks_, true, true);
    
    for (unsigned level = 0; level < eval_status.size(); level++)
    {
      if (!eval_status[level].success)
      {
        std::cerr << "ERROR: couldn't map level " << level_names.at(level) << ": "
                  << eval_status[level].fail_reason << std::endl;
        exit(1);
      }
    }
    // if (!std::accumulate(success.begin(), success.end(), true, std::logical_and<>{}))
    // {
    //   std::cout << "Illegal mapping, evaluation failed." << std::endl;
    //   return;
    // }

    std::cerr << "Utilization = " << engine.Utilization() << std::endl;
    std::cerr << "Sparse pJ / Dense MACC = "<< engine.SparseEnergy() / engine.GetTopology().MACCs() << std::endl;
    std::cerr << "Sparse Cycles = "<<engine.SparseCycles() << std::endl;
    std::cerr << "Speedup = " << static_cast<double>(engine.Cycles())/engine.SparseCycles() << std::endl;
    if(engine.GetDenseSuccess())
    {
      std::cerr << "Dense pJ/MACC = "<< engine.Energy() / engine.GetTopology().MACCs() << std::endl;
      std::cerr << "Energy Reduction = "<< std::setprecision(3) << engine.Energy() / engine.SparseEnergy() <<"x"<< std::endl;
    }
    else
    {
      std::cerr << "Dense Working Set Does Not Fit!" << std::endl;
      std::cerr << "Unknown Energy Reduction" << std::endl;
    }
      
    std::cout << std::endl;
    
    if (engine.IsEvaluated())
    {
      std::cout << "Utilization = " << std::setw(4) << std::fixed << std::setprecision(2) << engine.Utilization() 
                << " | Sparse pJ/MACC = " << std::setw(8) << std::fixed << std::setprecision(3) << engine.SparseEnergy() /
          engine.GetTopology().MACCs() << std::endl;
    
      std::ofstream map_txt_file(map_txt_file_name);
      mapping.PrettyPrint(map_txt_file, arch_specs_.topology.StorageLevelNames(), engine.GetTopology().TileSizes());
      map_txt_file.close();

      std::ofstream stats_file(stats_file_name);
      stats_file << engine << std::endl;
      stats_file.close();
    }

    // Print the engine stats and mapping to an XML file
    std::ofstream ofs(xml_file_name);
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(engine);
    ar << BOOST_SERIALIZATION_NVP(mapping);
    const Application* a = this;
    ar << BOOST_SERIALIZATION_NVP(a);
  }
};

