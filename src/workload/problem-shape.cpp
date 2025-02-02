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

#include "problem-shape.hpp"
#include "workload.hpp"
#include "operation-space.hpp"

namespace problem
{

// ======================================== //
//              Problem Shape               //
// ======================================== //

void Shape::Parse(config::CompoundConfigNode shape)
{
  // Not sure what to do with the name, since we only ever
  // parse one shape per invocation.
  name = "";
  shape.lookupValue("name", name);

  // Dimensions.
  config::CompoundConfigNode dimensions = shape.lookup("dimensions");
  assert(dimensions.isArray());

  NumDimensions = 0;
  std::vector<std::string> dim_names;
  shape.lookupArrayValue("dimensions", dim_names);
  for (const std::string& dim_name : dim_names )
  { 
    if (dim_name.length() != 1)
    {
      std::cerr << "ERROR: unfortunately, dimension names can only be 1 character in length. To remove this limitation, improve the constraint-parsing code in ParseUserPermutations() and ParseUserFactors() in mapping/parser.cpp and mapspaces/uber.hpp." << std::endl;
      exit(1);
    }
    DimensionIDToName[NumDimensions] = dim_name;
    DimensionNameToID[dim_name] = NumDimensions;
    NumDimensions++;
  }

  // Coefficients (optional). // strides and dialation, used as coefficient of SOP
  NumCoefficients = 0;
  if (shape.exists("coefficients"))
  {
    config::CompoundConfigNode coefficients = shape.lookup("coefficients");
    assert(coefficients.isList());
    for (int c = 0; c < coefficients.getLength(); c++)
    {
      auto coefficient = coefficients[c];
      std::string name;
      assert(coefficient.lookupValue("name", name));
           
      Coefficient default_value;
      assert(coefficient.lookupValue("default", default_value));

      CoefficientIDToName[NumCoefficients] = name;
      CoefficientNameToID[name] = NumCoefficients;
      DefaultCoefficients[NumCoefficients] = default_value;
      NumCoefficients++;
    }
  }
  
  // Data Spaces.
  config::CompoundConfigNode data_spaces = shape.lookup("data-spaces");
  assert(data_spaces.isList());

  NumDataSpaces = 0;
  for (int d = 0; d < data_spaces.getLength(); d++)
  {
    auto data_space = data_spaces[d];
    std::string name;
    assert(data_space.lookupValue("name", name));

    DataSpaceIDToName[NumDataSpaces] = name;
    DataSpaceNameToID[name] = NumDataSpaces;

    bool read_write = false;
    data_space.lookupValue("read-write", read_write);
    IsReadWriteDataSpace[NumDataSpaces] = read_write;

    Projection projection;
    config::CompoundConfigNode projection_cfg = data_space.lookup("projection");
    if (projection_cfg.isArray())
    {
      DataSpaceOrder[NumDataSpaces] = 0;
      std::vector<std::string> dim_names;
      data_space.lookupArrayValue("projection", dim_names);
      for (const std::string& dim_name : dim_names)
      {
        auto& dim_id = DimensionNameToID.at(dim_name);
        projection.push_back({{ NumCoefficients, dim_id }}); // simple       
        DataSpaceOrder[NumDataSpaces]++;
      }
    }
    else if (projection_cfg.isList())
    {
      DataSpaceOrder[NumDataSpaces] = 0;
      for (int k = 0; k < projection_cfg.getLength(); k++)
      {
        // Process one data-space dimension.
        ProjectionExpression expression;
        auto dimension = projection_cfg[k];
        // Each expression is a list of terms. Each term can be
        // a libconfig array or list.
        for (int t = 0; t < dimension.getLength(); t++)
        {
          auto term = dimension[t];
          assert(term.isArray());
          std::vector<std::string> nameAndCoeff;
          term.getArrayValue(nameAndCoeff);
          // Each term may have exactly 1 or 2 items.
          if (term.getLength() == 1)
          {
            const std::string& dim_name = nameAndCoeff[0];
            auto& dim_id = DimensionNameToID.at(dim_name);
            expression.push_back({ NumCoefficients, dim_id }); // simple
          }
          else if (term.getLength() == 2)
          {
            const std::string& dim_name = nameAndCoeff[0];
            const std::string& coeff_name = nameAndCoeff[1];
            auto& dim_id = DimensionNameToID.at(dim_name);
            auto& coeff_id = CoefficientNameToID.at(coeff_name);
            expression.push_back({ coeff_id, dim_id });
          }
          else
          {
            assert(false);
          }
        }
        
        projection.push_back(expression);
        DataSpaceOrder[NumDataSpaces]++;
      }
    }
    else
    {
      assert(false);
    }

    Projections.push_back(projection);
    NumDataSpaces++;
  }
}

}  // namespace problem
