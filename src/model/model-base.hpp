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
#include <iomanip>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace model
{

//
// Attribute.
//

template<class T>
class Attribute
{
 private:
  T t_; // value of the attributes
  std::string name_;
  bool specified_;
  // int which_const;

 public:
  Attribute() : t_(), name_("NONAME"), specified_(false)/*, which_const(0)*/ {}
  
  Attribute(T t) : t_(t), name_("NONAME"), specified_(true)/*, which_const(1)*/ {}

  Attribute(T t, std::string name) : t_(t), name_(name), specified_(true)/*, which_const(2)*/ {}
  
  bool IsSpecified() const { return specified_; }

  // int WhichConst() const { return which_const; }
  
  T Get() const
  {
    assert(specified_);
    return t_; // return value of the attributes
  }

  friend std::ostream& operator << (std::ostream& out, const Attribute& a)
  {
    if (a.specified_)
    {
      // FIXME: names aren't initialized properly.
      // out << std::left << std::setw(12) << a.name_;
      // out << " : ";
      out << a.t_;
    }
    else
    {
      out << "-";
    }
    return out;
  }

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      if (specified_)
        ar& BOOST_SERIALIZATION_NVP(t_);
    }
  }  
};

//
// Module.
//

class Module
{
 protected:
  bool is_specced_ = false;
  bool is_evaluated_ = false;

 public:
  bool IsSpecced() const { return is_specced_; }
  bool IsEvaluated() const { return is_evaluated_; }
};

} // namespace model
