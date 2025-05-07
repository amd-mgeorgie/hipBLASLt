/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include "UserDrivenTuningParser.hpp"
#include "hipblaslt_ostream.hpp"
#include <fstream>
#include <shared_mutex>
#include <sstream>
#include <utility>

namespace TensileLite
{

    void getContractionProblemsFromFile(const std::string& path)
    {
        OverrideMap&                m_override = OverrideMap::getMap();
        std::mutex&                 map_guard  = m_override.getLock();
        std::lock_guard<std::mutex> lock(map_guard);
        
        std::cout << "getContractionProblemsFromFile, m" << std::flush << std::endl;
        hipblaslt_cout << "getContractionProblemsFromFile, m" << std::endl;
        //log_info(__func__, "get")
        if(m_override.size() == 0)
        {
            std::cout << "getContractionProblemsFromFile, if" << std::flush << std::endl;
            hipblaslt_cout << "getContractionProblemsFromFile, if" << std::endl;

            std::ifstream file_read(path);
            std::string   line, entry;

            const auto verion      = "Git Version";
            const auto delim       = ',';
            const int  max_entries = 38;

            while(std::getline(file_read, line))
            {
                // Ignore lines without delimiter
                line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));

                std::cout << "getContractionProblemsFromFile, while" << std::flush << std::endl;
                hipblaslt_cout << "getContractionProblemsFromFile, while" << std::endl;

                if(line.find(delim) != std::string::npos && line.find(verion) == std::string::npos)
                {
                    std::cout << "getContractionProblemsFromFile, line.find" << std::flush << std::endl;
                    hipblaslt_cout << "getContractionProblemsFromFile, line.find" << std::endl;
                    std::vector<std::string> entries{};
                    entries.reserve(max_entries);

                    std::stringstream line_ss(line);
                    while(getline(line_ss, entry, delim))
                    {
                        entries.push_back(entry);
                    }
                    std::cout << "problemFromEntries, in" << std::flush << std::endl;
                    hipblaslt_cout << "problemFromEntries, in" << std::endl;
                    auto problemSolution = problemFromEntries(entries);
                    hipblaslt_cout << "Solution second index: " << problemSolution.second << std::endl;    
                    if(problemSolution.second > 0)
                    {
                        hipblaslt_cout << "Searching for, first " << std::endl;
                        auto sol_iter = m_override.find(problemSolution.first);
                        for(auto sol_idx = sol_iter.first; sol_idx != sol_iter.second; sol_idx++)
                        {
                            hipblaslt_cout << "Sol index second: " << sol_idx->second << std::endl;
                            hipblaslt_cout << "Problem solution second: " << problemSolution.second << std::endl;
                            
                            if(sol_idx->second == problemSolution.second)
                            {
                                hipblaslt_cout << "Duplicate found, erased! " << std::endl;
                                m_override.erase(sol_idx);
                                break;
                            }
                        }

                        hipblaslt_cout << "Solution added!" << std::endl;
                        m_override.add(problemSolution);
                    }
                }
            }
        }
    }

    std::pair<ProblemOverride, int> problemFromEntries(const std::vector<std::string>& entries)
    {
        const size_t entries_n = entries.size();

        std::cout << "problemFromEntries, m" << std::flush << std::endl;
        hipblaslt_cout << "problemFromEntries, m" << std::endl;
        hipblaslt_cout << "Entries: " << entries_n << std::endl;
        //log_info(__func__, "problemFromEntries, m");
        if(entries_n != 38)
        {
            std::cout << "Wrong number of indices, m" << std::flush << std::endl;
            hipblaslt_cout << "Wrong number of indices, m" << std::endl;
            // log_info(__func__, "Wrong number of indices, m");
            return std::make_pair(ProblemOverride{}, -1);
        }

        //TODO: not correct expected format!
        // Expected format: transA,transB, batch_count, M,N,K,input_type,output_type,compute_type,solution_index
        // Example, 37 positions (position index at second row)

        // transA, transB, grouped_gemm, batch_count,    m,     n,   k, alpha, lda, stride_a, beta, ldb, stride_b,  ldc,   stride_c,  ldd,  stride_d, a_type, b_type, c_type, d_type, compute_type, scaleA, scaleB, scaleC, scaleD, amaxD, activation_type, bias_vector, bias_type, aux_type, rotating_buffer, hipblaslt-Gflops, hipblaslt-GB/s,      us,  sol_idx, arch, CUs
        //      T,      N,            0,           1, 3456, 70000, 512,     1, 512,  1769472,    0, 512, 35840000, 3456,  241920000, 3456, 241920000,  f32_r,  f32_r,  f32_r,  f32_r,       xf32_r,      0,      0,      0,      0,     0,            none,           0,     f32_r,    f32_r,             512,           232946,        979.201, 1063.45
        //      0,      1,            2,           3,    4,     5,   6,     7,   8,        9,   10,  11,       12,   13,         14,   15,        16,     17,     18,     19,     20,           21,     22,     23,     24,     25,    26,              27,          28,        29,       30,              31,               32,             33,      34,      35 ,   36,   37

        bool transA = (entries[0] != "N");
        bool transB = (entries[1] != "N");

        size_t           m, n, b, k;
        rocisa::DataType inputTypeA  = rocisa::DataType::None;
        rocisa::DataType inputTypeB  = rocisa::DataType::None;
        rocisa::DataType outputType  = rocisa::DataType::None;
        rocisa::DataType biasType    = rocisa::DataType::None;

        rocisa::DataType computeType = rocisa::DataType::None;

        bool to_use_bias = false;
        int rotSize = 0;
        int solution_idx = -1;

        try
        {
            // TODO: are any additional mapping parameters needed?
            // bias use has been added and rotating buffer
            // TOREM
            hipblaslt_cout << "Fetched Parameters:" << std::endl;

            b            = std::stol(entries[3]);
            hipblaslt_cout << "b: " << b << std::endl;

            m            = std::stol(entries[4]);
            hipblaslt_cout << "m: " << m << std::endl;

            n            = std::stol(entries[5]);
            hipblaslt_cout << "n: " << n << std::endl;
            
            k            = std::stol(entries[6]);
            hipblaslt_cout << "k: " << k << std::endl;

            inputTypeA   = hipDataType_to_tensile_type(string_to_hip_datatype(entries[17]));
            hipblaslt_cout << "Input Type A: " << inputTypeA << std::endl;
            
            inputTypeB   = hipDataType_to_tensile_type(string_to_hip_datatype(entries[18]));
            hipblaslt_cout << "Input Type B: " << inputTypeB << std::endl;
            
            outputType   = hipDataType_to_tensile_type(string_to_hip_datatype(entries[19]));
            hipblaslt_cout << "Output Type: " << outputType << std::endl;
            
            biasType = hipDataType_to_tensile_type(string_to_hip_datatype(entries[29]));
            hipblaslt_cout << "Bias type " << biasType << std::endl;
            
            computeType  = hipDataType_to_tensile_type(string_to_hip_datatype(entries[21]));
            hipblaslt_cout << "Compute Type: " << computeType << std::endl;
            
            to_use_bias  = static_cast<bool>(std::abs(std::stoi(entries[28])));
            hipblaslt_cout << "Use To Bias: " << to_use_bias << std::endl;

            rotSize  = std::abs(std::stoi(entries[31]));
            hipblaslt_cout << "Rotating Buffer: " << rotSize << std::endl;

            // computeType  = hipDataType_to_tensile_compute_type(string_to_hipblas_computetype(entries[21]));
            // TODO: Workaround for TF32 support, should be handled with the above
            
            if (entries[21] == "xf32_r")
                computeType = rocisa::DataType::XFloat32;
                                    
            solution_idx = std::stoi(entries[35]);

            hipblaslt_cout << "Solution Index: " << solution_idx << std::endl;                        
        }
        catch(std::invalid_argument const& ex)
        {
            hipblaslt_cout << "Invalid argument" << std::endl;
            return std::make_pair(ProblemOverride{}, -1);
        }
        catch(std::out_of_range const& ex)
        {
            hipblaslt_cout << "out of range" << std::endl;
            return std::make_pair(ProblemOverride{}, -1);
        }

        if(inputTypeA == rocisa::DataType::None || inputTypeB == rocisa::DataType::None
           || outputType == rocisa::DataType::None || computeType == rocisa::DataType::None)
        {
            hipblaslt_cout << "all None" << std::endl;
            return std::make_pair(ProblemOverride{}, -1);
        }
        hipblaslt_cout << "Problem po, in" << std::endl;
        ProblemOverride po(
            transA, transB, inputTypeA, inputTypeB, computeType, outputType, biasType, m, n, k, b);
        
        hipblaslt_cout << "Output type: " << po.outputType() << std::endl;
        hipblaslt_cout << "Bias type: " << po.biasType() << std::endl;
        hipblaslt_cout << "Use bias:"  << po.to_use_bias() << std::endl;
        hipblaslt_cout << "Rot buffer:" << po.rotSize()  << std::endl;    
        return std::make_pair(po, solution_idx);
    }

    ProblemOverride::ProblemOverride()
        : m_transA(false)
        , m_transB(false)
        , m_inputTypeA(rocisa::DataType::None)
        , m_inputTypeB(rocisa::DataType::None)
        , m_computeType(rocisa::DataType::None)
        , m_outputType(rocisa::DataType::None)
        , m_biasType(rocisa::DataType::None)
        , m_m(0)
        , m_n(0)
        , m_k(0)
        , m_batchSize(0)
        , m_to_use_bias(false)
        , m_rotSize(0)
    {
        
    }

    ProblemOverride::ProblemOverride(bool             transA,
                                     bool             transB,
                                     rocisa::DataType inputTypeA,
                                     rocisa::DataType inputTypeB,
                                     rocisa::DataType computeType,
                                     rocisa::DataType outputType,
                                     rocisa::DataType biasType,
                                     size_t           m,
                                     size_t           n,
                                     size_t           k,
                                     size_t           batchSize)
        : m_transA(transA)
        , m_transB(transB)
        , m_inputTypeA(inputTypeA)
        , m_inputTypeB(inputTypeB)
        , m_computeType(computeType)
        , m_outputType(outputType)
        , m_biasType(biasType)
        , m_m(m)
        , m_n(n)
        , m_k(k)
        , m_batchSize(batchSize)        
        , m_to_use_bias(false)
        , m_rotSize(0)
    {
    }

    ProblemOverride::ProblemOverride(const ProblemOverride& problem)
    {

        m_transA      = problem.transA();
        m_transB      = problem.transB();
        m_inputTypeA  = problem.inputTypeA();
        m_inputTypeB  = problem.inputTypeB();
        m_computeType = problem.computeType();
        m_outputType  = problem.outputType();
        m_biasType    = problem.biasType();
        m_m           = problem.m();
        m_n           = problem.n();
        m_k           = problem.k();
        m_batchSize   = problem.batchSize();        
        m_to_use_bias = problem.to_use_bias();
        m_rotSize     = problem.rotSize();
    }
};
