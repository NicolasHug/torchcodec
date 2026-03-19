// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "Benchmark.h"

namespace facebook::torchcodec {

namespace {
std::atomic<bool> g_benchmarkEnabled{false};
} // namespace

void BenchmarkData::record(const std::string& name, double ms) {
  std::lock_guard<std::mutex> lock(mutex);
  auto& entry = entries[name];
  entry.totalTimeMs += ms;
  entry.count++;
}

std::map<std::string, BenchmarkData::Entry> BenchmarkData::getResults() {
  std::lock_guard<std::mutex> lock(mutex);
  return entries;
}

void BenchmarkData::reset() {
  std::lock_guard<std::mutex> lock(mutex);
  entries.clear();
}

BenchmarkData& getBenchmarkData() {
  static BenchmarkData data;
  return data;
}

bool isBenchmarkEnabled() {
  return g_benchmarkEnabled.load(std::memory_order_relaxed);
}

void setBenchmarkEnabled(bool enabled) {
  g_benchmarkEnabled.store(enabled, std::memory_order_relaxed);
}

ScopedBenchmarkTimer::ScopedBenchmarkTimer(const char* name)
    : name_(name),
      start_(std::chrono::high_resolution_clock::now()),
      enabled_(isBenchmarkEnabled()) {}

ScopedBenchmarkTimer::~ScopedBenchmarkTimer() {
  if (!enabled_) {
    return;
  }
  auto end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start_).count();
  getBenchmarkData().record(name_, ms);
}

} // namespace facebook::torchcodec
