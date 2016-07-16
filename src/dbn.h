/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-16

*/

#ifndef DBN_H
#define DBN_H

#include <vector>
#include "rbm.h"
#include "mlp.h"
#include "./math/matrix.h"

class DBN {
public:
  DBN() = default;
  DBN(const std::vector<int> &);
  DBN(const std::initializer_list<int> &);
  DBN(const DBN &) = default;
  DBN(DBN &&) = default;
  DBN & operator=(const DBN &) = default;
  DBN & operator=(DBN &&) = default;
  ~DBN() = default;
  void pre_train(const Matrix &, int, double, int);
  void fine_tune(const Matrix &, const Matrix &, double, double);
  Matrix predict(const Matrix &);
private:
  std::vector<RBM> rbms;
  MLP mlp;
};

#endif  // DBN.h