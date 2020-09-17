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

#include "workload.hpp"
#include "data-space.hpp"
#include "per-data-space.hpp"

namespace problem
{

// ======================================== //
//              OperationPoint              //
// ======================================== //

class OperationPoint : public Point
{
 public:
  OperationPoint() :
      Point(GetShape()->NumDimensions) // init with order (7 CNN)
  {
  }

  OperationPoint operator + (const OperationPoint& d)
  {
    OperationPoint retval;
    for(unsigned i=0;i<d.Order();i++)
      retval[i] = this->coordinates_[i]+d.coordinates_[i];
    return retval;
  }

  OperationPoint& operator += (const OperationPoint& d)
  {
    for(unsigned i=0;i<d.Order();i++)
      this->coordinates_[i] += d.coordinates_[i];

    return (*this);
  }

  // modified from OperationSpace class, this is a more restricted version that only convert to DataPoint 
  // which currently used for weight Only, see how we got d
  DataPoint Project(const Workload* wc) const
  {
    DataPoint retval;

    auto d = GetShape()->name == "Weight-Update-Depthwise" ? 
              (GetShape()->DataSpaceNameToID.at("Inputs")) : 
              (GetShape()->DataSpaceNameToID.at("Weights"));
    for (unsigned dim = 0; dim < retval.Order(); dim++)
    {
      retval[dim] = 0;
      for (auto & term : wc->GetShape()->Projections.at(d).at(dim)) // ProjectionTerm : ProjectionExpression
      { 
        Coordinate x = this->coordinates_[term.second]; // Coordinate is element in Point
        // FIXME: somehow "compile" the coefficients down for a given
        // workload config so that we avoid the branch and lookup below.
        if (term.first != wc->GetShape()->NumCoefficients) // linear combination
          retval[dim] += (x * wc->GetCoefficient(term.first));
        else // simple
          retval[dim] += x;
      }
    }

    return retval;
  }

  std::size_t Volume()
  {
    std::size_t volume=1;
    for(const auto &coordinate : this->coordinates_)
      volume *= coordinate;
    return volume;
  }
};

// ======================================== //
//              OperationSpace              //
// ======================================== //

class OperationSpace
{
 private:
  const Workload* workload_;

  std::vector<DataSpace> data_spaces_; // DataSpace is PointSet (impl. as AAHR)

 private:
  Point Project(Shape::DataSpaceID d, const Workload* wc,
                const OperationPoint& problem_point);
  
 public:
  OperationSpace();
  OperationSpace(const Workload* wc);
  OperationSpace(const Workload* wc, const OperationPoint& low,
                 const OperationPoint& high);

  void Reset();
  OperationSpace& operator+=(const OperationSpace& s);
  OperationSpace& operator+=(const OperationPoint& p);
  OperationSpace& ExtrudeAdd(const OperationSpace& s);
  OperationSpace operator-(const OperationSpace& p);
  DataSpace& GetDataSpace(Shape::DataSpaceID pv);
  PerDataSpace<std::size_t> GetSizes() const;
  std::size_t GetSize(const int t) const;
  DataSpace GetDataSpace(const int t) const;
  bool IsEmpty(const int t) const;
  bool CheckEquality(const OperationSpace& rhs, const int t) const;
  void PrintSizes();
  void Print(std::ostream& out = std::cerr) const;
  void Print(Shape::DataSpaceID pv, std::ostream& out = std::cerr) const;
};

std::ostream& operator << (std::ostream& out, const OperationSpace& os);

} // namespace problem
