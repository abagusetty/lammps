// clang-format off
#ifndef SYCL_FEW_H
#define SYCL_FEW_H

#include <Sycl_Core.hpp>

template <typename T, std::size_t n>
class Few {
  alignas(T) char array_[n * sizeof(T)];

 public:
  enum { size = n };
  Few(std::initializer_list<T> l) {
    std::size_t i = 0;
    for (auto it = l.begin(); it != l.end(); ++it) {
      new (data() + (i++)) T(*it);
    }
  }
  __attribute__((always_inline)) Few(T const a[]) {
    for (std::size_t i = 0; i < n; ++i) new (data() + i) T(a[i]);
  }
  __attribute__((always_inline)) Few() {
    for (std::size_t i = 0; i < n; ++i) new (data() + i) T();
  }
  __attribute__((always_inline)) ~Few() {
    for (std::size_t i = 0; i < n; ++i) (data()[i]).~T();
  }
  __attribute__((always_inline)) Few(Few<T, n> const& rhs) {
    for (std::size_t i = 0; i < n; ++i) new (data() + i) T(rhs[i]);
  }
  __attribute__((always_inline)) Few(Few<T, n> const volatile& rhs) {
    for (std::size_t i = 0; i < n; ++i) new (data() + i) T(rhs[i]);
  }
  __attribute__((always_inline)) void operator=(Few<T, n> const& rhs) volatile {
    for (std::size_t i = 0; i < n; ++i) data()[i] = rhs[i];
  }
  __attribute__((always_inline)) void operator=(Few<T, n> const& rhs) {
    for (std::size_t i = 0; i < n; ++i) data()[i] = rhs[i];
  }
  __attribute__((always_inline)) void operator=(Few<T, n> const volatile& rhs) {
    for (std::size_t i = 0; i < n; ++i) data()[i] = rhs[i];
  }
  __attribute__((always_inline)) T* data() {
    return reinterpret_cast<T*>(array_);
  }
  __attribute__((always_inline)) T const* data() const {
    return reinterpret_cast<T const*>(array_);
  }
  __attribute__((always_inline)) T volatile* data() volatile {
    return reinterpret_cast<T volatile*>(array_);
  }
  __attribute__((always_inline)) T const volatile* data() const volatile {
    return reinterpret_cast<T const volatile*>(array_);
  }
  __attribute__((always_inline)) T& operator[](std::size_t i) {
    return data()[i];
  }
  __attribute__((always_inline)) T const& operator[](std::size_t i) const {
    return data()[i];
  }
  __attribute__((always_inline)) T volatile& operator[](std::size_t i) volatile {
    return data()[i];
  }
  __attribute__((always_inline)) T const volatile& operator[](std::size_t i) const volatile {
    return data()[i];
  }
};

#endif
