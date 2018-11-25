/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * Written (W) 2014 Khaled Nasr
 */

#include "shogun/neuralnets/NeuralRectifiedLinearLayer.h"
#include "shogun/neuralnets/NeuralInputLayer.h"
#include "shogun/lib/SGVector.h"
#include "shogun/lib/SGMatrix.h"
#include "shogun/mathematics/Math.h"
#include "gtest/gtest.h"

#include <tuple>
#include <memory>

using namespace shogun;

class NeuralRectifiedLinearLayerTest  : public ::testing::Test {
 public:

  template <typename T>
  SGMatrix<T> CreateRandSGMat(int32_t num_features, int32_t num_samples,
  		T lower_bound, T upper_bound) {
	  auto data_batch = SGMatrix<T>(num_features, num_samples);
	  for (int32_t i=0; i<data_batch.num_rows*data_batch.num_cols; i++) {
		  data_batch[i] = CMath::random(lower_bound, upper_bound);
	  }
	  return data_batch;
  }

  template <typename T>
  auto CreateRandSGMatAndInputLayer(
  		int32_t num_features, int32_t num_samples,
  		T lower_bound, T upper_bound) {
	  auto data_batch = CreateRandSGMat(num_features, num_samples, lower_bound, upper_bound);
	  // Can't use unique_ptr here due to "unref_all"
	  auto input_layer = new CNeuralInputLayer(data_batch.num_rows);
	  input_layer->set_batch_size(data_batch.num_cols);
	  input_layer->compute_activations(data_batch);
	  return std::make_tuple(data_batch, input_layer);
  }

  auto InitNeuralLinearLayer(CNeuralLinearLayer* layer,
  		CDynamicObjectArray* layers, SGVector<int32_t> input_indices,
  		int32_t batch_size, double sigma) {
	  layer->initialize_neural_layer(layers, input_indices);
	  SGVector<float64_t> params(layer->get_num_parameters());
	  SGVector<bool> param_regularizable(layer->get_num_parameters());
	  layer->initialize_parameters(params, param_regularizable, sigma);
	  layer->set_batch_size(batch_size);
	  layer->compute_activations(params, layers);
	  return params;
  }
 protected:
  void SetUp() {
	  CMath::init_random(100);
	  m_layers = std::make_unique<CDynamicObjectArray>();
  }
  std::unique_ptr<CDynamicObjectArray> m_layers;
};

/** Compares the activations computed using the layer against manually computed
 * activations
 */
TEST_F(NeuralRectifiedLinearLayerTest, compute_activations)
{

	// initialize some random inputs
	SGMatrix<float64_t> x;
	CNeuralInputLayer* input;
	std::tie(x, input) = CreateRandSGMatAndInputLayer<float64_t>(12, 3, -10.0, 10.0);
	m_layers->append_element(input);

	// initialize the layer
	CNeuralRectifiedLinearLayer layer(9);
	SGVector<int32_t> input_indices(1);
	input_indices[0] = 0;
	auto params = InitNeuralLinearLayer(&layer, m_layers.get(), input_indices, x.num_cols, 1.0);
	SGMatrix<float64_t> A = layer.get_activations();

	// manually compute the layer's activations
	SGMatrix<float64_t> A_ref(layer.get_num_neurons(), x.num_cols);

	float64_t* biases = params.vector;
	float64_t* weights = biases + layer.get_num_neurons();

	for (int32_t i=0; i<A_ref.num_rows; i++)
	{
		for (int32_t j=0; j<A_ref.num_cols; j++)
		{
			A_ref(i,j) = biases[i];

			for (int32_t k=0; k<x.num_rows; k++)
				A_ref(i,j) += weights[i+k*A_ref.num_rows]*x(k,j);

			A_ref(i,j) = CMath::max<float64_t>(0, A_ref(i,j));
		}
	}

	// compare
	EXPECT_EQ(A_ref.num_rows, A.num_rows);
	EXPECT_EQ(A_ref.num_cols, A.num_cols);
	for (int32_t i=0; i<A.num_rows*A.num_cols; i++)
		EXPECT_NEAR(A_ref[i], A[i], 1e-12);
}

/** Compares the parameter gradients computed using the layer, when the layer
 * is used as a hidden layer, against gradients computed using numerical
 * approximation
 */
TEST_F(NeuralRectifiedLinearLayerTest, compute_parameter_gradients_hidden) {
	
	// initialize some random inputs
	SGMatrix<float64_t> x1, x2;
	CNeuralInputLayer *input1, *input2;
	std::tie(x1, input1) = CreateRandSGMatAndInputLayer<float64_t>(12, 3, -10.0, 10.0);
	std::tie(x2, input2) = CreateRandSGMatAndInputLayer<float64_t>(7, 3, -10.0, 10.0);
	m_layers->append_element(input1);
	m_layers->append_element(input2);

	// initialize the hidden layer
	CNeuralLinearLayer* layer_hid = new CNeuralRectifiedLinearLayer(5);
	m_layers->append_element(layer_hid);
	SGVector<int32_t> input_indices_hid(2);
	input_indices_hid[0] = 0;
	input_indices_hid[1] = 1;
	auto param_hid = InitNeuralLinearLayer(layer_hid, m_layers.get(), input_indices_hid, x1.num_cols, 0.01);

	// initialize the output layer
	auto y = CreateRandSGMat(9, 3, 1.0, 0.0);
	CNeuralLinearLayer layer_out(y.num_rows);
	SGVector<int32_t> input_indices_out(1);
	input_indices_out[0] = 2;
	auto param_out = InitNeuralLinearLayer(&layer_out, m_layers.get(), input_indices_out, x1.num_cols, 0.01);

	// compute gradients
	layer_hid->get_activation_gradients().zero();
	SGVector<float64_t> gradients_out(layer_out.get_num_parameters());
	layer_out.compute_gradients(param_out, y, m_layers.get(), gradients_out);

	SGVector<float64_t> gradients_hid(layer_hid->get_num_parameters());
	layer_hid->compute_gradients(param_hid, SGMatrix<float64_t>(),
			m_layers.get(), gradients_hid);

	// manually compute parameter gradients
	SGVector<float64_t> gradients_hid_numerical(layer_hid->get_num_parameters());
	float64_t epsilon = 1e-9;
	for (int32_t i=0; i<layer_hid->get_num_parameters(); i++)
	{
		param_hid[i] += epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer_hid->compute_activations(param_hid, m_layers.get());
		layer_out.compute_activations(param_out, m_layers.get());
		float64_t error_plus = layer_out.compute_error(y);

		param_hid[i] -= 2*epsilon;
		input1->compute_activations(x1);
		input2->compute_activations(x2);
		layer_hid->compute_activations(param_hid, m_layers.get());
		layer_out.compute_activations(param_out, m_layers.get());
		float64_t error_minus = layer_out.compute_error(y);
		param_hid[i] += epsilon;

		gradients_hid_numerical[i] = (error_plus-error_minus)/(2*epsilon);
	}

	// compare
	for (int32_t i=0; i<gradients_hid_numerical.vlen; i++)
		EXPECT_NEAR(gradients_hid_numerical[i], gradients_hid[i], 1e-6);
}
