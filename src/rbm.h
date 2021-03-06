/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-15
Last modified: 2016-07-16

*/

#ifndef RBM_H
#define RBM_H

#include "./math/matrix.h"

class RBM {
public:
  RBM() = default;
  // RBM(int, int);  // specificaly for DBN
  RBM(Matrix &, const Matrix &, Matrix &);  // specificaly for DBN, const is omitted in order to share the weight
  RBM(const RBM &) = default;
  RBM(RBM &&) = default;
  RBM & operator=(const RBM &) = default;
  RBM & operator=(RBM &&) = default;
  Matrix reconstruct(const Matrix &);
  void train(const Matrix &, double, int);
  Matrix prop_vh(const Matrix &);  // Specifically for DBM, turn private to be public
private:
  Matrix & weight;
  Matrix w_bias_vis;
  Matrix & w_bias_hid;
  // Matrix prop_vh(const Matrix &);
  Matrix prop_hv(const Matrix &);
  Matrix sample_vh(const Matrix &);
  Matrix sample_hv(const Matrix &);
  Matrix gibbs_vhv(const Matrix &);
  Matrix gibbs_hvh(const Matrix &);
  void contrastive_divergence(const Matrix &, double, int);
};

#endif  // rbm.h