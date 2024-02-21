/*
 * Copyright (c) 2020, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: David Fridovich-Keil   ( dfk@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// Additional test for ExtremeValueCost..
//
///////////////////////////////////////////////////////////////////////////////

#include <ilqgames/cost/extreme_value_cost.h>
#include <ilqgames/cost/quadratic_cost.h>
#include <ilqgames/cost/signed_distance_cost.h>
#include <ilqgames/utils/types.h>

#include <gtest/gtest.h>
#include <math.h>
#include <memory>

using namespace ilqgames;

// Check that evaluation is consistent.
TEST(ExtremeValueCostTest, EvaluatesConsistentCost) {
  const std::shared_ptr<const SignedDistanceCost> cost1(
      new SignedDistanceCost({0, 1}, {2, 3}, 5.0));
  const std::shared_ptr<const QuadraticCost> cost2(
      new QuadraticCost(1.0, -1, 1.0));
  const ExtremeValueCost cost({cost1, cost2}, true);

  float evaluated;
  const VectorXf input = VectorXf::Random(4);
  const Cost* active = cost.ExtremeCost(0.0, input, &evaluated);
  EXPECT_LT(std::abs(active->Evaluate(0.0, input) - evaluated),
            constants::kSmallNumber);
}
