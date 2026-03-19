// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>

namespace facebook::torchcodec {

struct BenchmarkData {
  struct Entry {
    double totalTimeMs = 0.0;
    int64_t count = 0;
  };

  std::mutex mutex;
  std::map<std::string, Entry> entries;

  void record(const std::string& name, double ms);
  std::map<std::string, Entry> getResults();
  void reset();
};

BenchmarkData& getBenchmarkData();
bool isBenchmarkEnabled();
void setBenchmarkEnabled(bool enabled);

// RAII timer for CPU-side operations
class ScopedBenchmarkTimer {
 public:
  explicit ScopedBenchmarkTimer(const char* name);
  ~ScopedBenchmarkTimer();

 private:
  const char* name_;
  std::chrono::high_resolution_clock::time_point start_;
  bool enabled_;
};

#define BENCHMARK_TIMER(name) \
  ScopedBenchmarkTimer _bench_timer_##__LINE__(name)

} // namespace facebook::torchcodec
