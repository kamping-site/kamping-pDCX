/*******************************************************************************
 * mpi/distribute_input.hpp
 *
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <tlx/digest/sha1.hpp>
#include <vector>

#include "mpi/broadcast.hpp"
#include "mpi/environment.hpp"
#include "mpi/scan.hpp"
#include "mpi/shift.hpp"
#include "mpi/type_mapper.hpp"
#include "util/string.hpp"
#include "util/string_set.hpp"

namespace dsss::mpi {

static dsss::distributed_string distribute_string(
    const std::string& input_path, size_t max_size = 0,
    environment env = environment()) {
  MPI_File mpi_file;

  MPI_File_open(env.communicator(),
                (char*)input_path.c_str(),  // ugly cast to use old C interface
                MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

  MPI_Offset global_file_size = 0;
  MPI_File_get_size(mpi_file, &global_file_size);
  if (max_size > 0) {
    global_file_size =
        std::min(max_size, static_cast<size_t>(global_file_size));
  }

  size_t local_slice_size = global_file_size / env.size();
  int64_t larger_slices = global_file_size % env.size();

  size_t offset;
  if (env.rank() < larger_slices) {
    ++local_slice_size;
    offset = local_slice_size * env.rank();
  } else {
    offset = larger_slices * (local_slice_size + 1);
    offset += (env.rank() - larger_slices) * local_slice_size;
  }

  MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

  std::vector<dsss::char_type> result(local_slice_size);

  MPI_File_read(mpi_file, result.data(), local_slice_size,
                type_mapper<dsss::char_type>::type(), MPI_STATUS_IGNORE);

  return dsss::distributed_string{offset, result};
}

dsss::distributed_string distribute_strings(const std::string& input_path,
                                            size_t max_size = 0,
                                            environment env = environment()) {
  MPI_File mpi_file;

  MPI_File_open(env.communicator(),
                (char*)input_path.c_str(),  // ugly cast to use old C interface
                MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

  MPI_Offset global_file_size = 0;
  MPI_File_get_size(mpi_file, &global_file_size);
  if (max_size > 0) {
    global_file_size =
        std::min(max_size, static_cast<size_t>(global_file_size));
  }

  size_t local_slice_size = global_file_size / env.size();
  int64_t larger_slices = global_file_size % env.size();

  size_t offset;
  if (env.rank() < larger_slices) {
    ++local_slice_size;
    offset = local_slice_size * env.rank();
  } else {
    offset = larger_slices * (local_slice_size + 1);
    offset += (env.rank() - larger_slices) * local_slice_size;
  }

  MPI_File_seek(mpi_file, offset, MPI_SEEK_SET);

  std::vector<dsss::char_type> result(local_slice_size);

  MPI_File_read(mpi_file, result.data(), local_slice_size,
                type_mapper<dsss::char_type>::type(), MPI_STATUS_IGNORE);

  size_t first_end = 0;
  while (first_end < result.size() && result[first_end] != 0) {
    ++first_end;
  }

  std::vector<dsss::char_type> end_of_last_string =
      dsss::mpi::shift_left(result.data(), first_end + 1);
  // We copy this string, even if it's not the end of the last one on the
  // previous PE, but a new string. This way, we can delete it on the sending
  // PE without checking if it was the end.
  if (env.rank() + 1 < env.size()) {
    std::copy_n(end_of_last_string.begin(), end_of_last_string.size(),
                std::back_inserter(result));
  } else {
    if (result.back() != 0) {
      result.emplace_back(0);
    }  // Make last string end
  }
  if (env.rank() > 0) {  // Delete the sent string
    result.erase(result.begin(), result.begin() + first_end + 1);
  }

  return dsss::distributed_string{offset, result};
}

template <typename DataType>
static void write_data(std::vector<DataType>& local_data,
                       const std::string file_name,
                       environment env = environment()) {
  MPI_File mpi_file;

  MPI_File_open(env.communicator(), const_cast<char*>(file_name.c_str()),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file);

  MPI_File_write_ordered(mpi_file, local_data.data(),
                         local_data.size() * sizeof(DataType), MPI_BYTE,
                         MPI_STATUS_IGNORE);

  MPI_File_close(&mpi_file);
}

template <typename DataType>
static std::vector<DataType> read_data(const std::string file_name,
                                       environment env = environment()) {
  MPI_File mpi_file;

  MPI_File_open(env.communicator(), const_cast<char*>(file_name.c_str()),
                MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

  MPI_Offset global_file_size = 0;
  MPI_File_get_size(mpi_file, &global_file_size);
  global_file_size /= sizeof(DataType);

  size_t local_slice_size = global_file_size / env.size();
  int64_t larger_slices = global_file_size % env.size();

  size_t offset;
  if (env.rank() < larger_slices) {
    ++local_slice_size;
    offset = local_slice_size * env.rank();
  } else {
    offset = larger_slices * (local_slice_size + 1);
    offset += (env.rank() - larger_slices) * local_slice_size;
  }

  MPI_File_seek(mpi_file, offset * sizeof(DataType), MPI_SEEK_SET);

  std::vector<DataType> result(local_slice_size);

  MPI_File_read(mpi_file, result.data(), local_slice_size * sizeof(DataType),
                MPI_BYTE, MPI_STATUS_IGNORE);

  return result;
}

}  // namespace dsss::mpi

/******************************************************************************/
