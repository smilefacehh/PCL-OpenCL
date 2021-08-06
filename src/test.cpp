#include <iostream>
#include <chrono>

// Eigen
#include <eigen3/Eigen/Core>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

// opencv
#include <opencv2/opencv.hpp>

// opencl
#include <CL/cl.h>

/**
 * opencl版本
*/
void cl_depth2poincloud(const cv::Mat& rgb, const cv::Mat& depth, const Eigen::Matrix3d& intrinsic, const float& depth_factor, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointcloud)
{
    int w = rgb.cols, h = rgb.rows;

    cl_int status;
    cl_uint platform_N;
    cl_platform_id* platform_ids;
    cl_device_id device;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    
    // 1.获取平台信息
    status = clGetPlatformIDs(0, NULL, &platform_N);
    if(status != CL_SUCCESS)
    {
        std::cout << "错误：无法获取平台信息 " << status << std::endl;
        return;
    }

    if(platform_N <= 0)
    {
        std::cout << "错误：平台数量为0" << std::endl;
        return;
    }

    platform_ids = (cl_platform_id*) alloca(sizeof(cl_platform_id) * platform_N);
    status = clGetPlatformIDs(platform_N, platform_ids, NULL);
    if(status != CL_SUCCESS)
    {
        std::cout << "错误：无法获取平台信息 " << status << std::endl;
        return;
    }

    // 打印所有平台信息
    if(0)
    {
        char tmpName[40];
        for (size_t i = 0; i < platform_N; i++)
        {
            status = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(tmpName), &tmpName, NULL);
            std::cout << "platform " << i <<" name: " << tmpName << std::endl;
        
            status = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, sizeof(tmpName), &tmpName, NULL);
            std::cout << "platform " << i << " vendor: " << tmpName << std::endl;
        
            status = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, sizeof(tmpName), &tmpName, NULL);
            std::cout << "platform " << i << " version: " << tmpName << std::endl;
        
            status = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_PROFILE, sizeof(tmpName), &tmpName, NULL);
            std::cout << "platform " << i << " profile: " << tmpName << std::endl;
        }
    }

    // 2.获取当前平台
#if defined(__x86_64__) || defined(__x86_64) || defined(_M_AMD64) || defined(_M_X64)
    status = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
#elif defined(__arm__) || defined(__ARMEL__) || defined(_M_ARM)
    status = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
#endif

    if(status != CL_SUCCESS)
    {
        std::cout << "错误：无法获取平台信息 " << status << std::endl;
        return;
    }

    // 打印当前平台信息
    if(0)
    {
        std::cout << "*******设备信息*******" << std::endl;
        cl_uint maxComputeUnits;
        status = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
        std::cout << "compute units: " << maxComputeUnits << std::endl;

        cl_uint maxWorkItemDim;
        status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDim), &maxWorkItemDim, NULL);
        std::cout << "max work item dimensions: " << maxWorkItemDim << std::endl;

        size_t maxWorkItemPerGroup;
        status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(maxWorkItemPerGroup), &maxWorkItemPerGroup, NULL);
        std::cout << "max workitem per group: " << maxWorkItemPerGroup << std::endl;

        cl_ulong maxGlobalMemSize;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(maxGlobalMemSize), &maxGlobalMemSize, NULL);
        std::cout << "max global mem: " << maxGlobalMemSize / 1024 / 1024 << "MB" << std::endl;

        cl_ulong maxConstantBufferSize;
        clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,sizeof(maxConstantBufferSize), &maxConstantBufferSize, NULL);
        std::cout << "max constant mem: " << maxConstantBufferSize / 1024 << "KB" << std::endl;

        cl_ulong maxLocalMemSize;
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,sizeof(maxLocalMemSize), &maxLocalMemSize, NULL);
        std::cout << "max local mem: " << maxLocalMemSize / 1024 << "KB" << std::endl;
    }

    // 3.创建上下文
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if(!context || status != CL_SUCCESS)
    {
        std::cout << "错误：创建上下文失败 " << status << std::endl;
        status = CL_INVALID_CONTEXT;
        return;
    }

    // 4.创建命令队列
    command_queue = clCreateCommandQueue(context, device, 0, &status);
    if(!command_queue || status != CL_SUCCESS)
    {
        std::cout << "错误：创建命令队列失败 " << status << std::endl;
        status = CL_INVALID_COMMAND_QUEUE;
        return;
    }
    
    // 5.加载kernel
    std::ifstream kernelfs("../kernel/depth2pointcloud.cl");
    if(!kernelfs.is_open())
    {
        std::cout << "错误：加载depth2pointcloud.cl文件失败，检查路径" << std::endl;
        return;
    }
    std::stringstream kernelss;
    kernelss << kernelfs.rdbuf();
    std::string code(kernelss.str());

    // 6.创建程序
    char* code_str = const_cast<char*>(code.c_str());
    size_t sz = code.size();
    program = clCreateProgramWithSource(context, 1, (const char**)&code_str, (const size_t*)&sz, &status);
    if(!program || status != CL_SUCCESS)
    {
        std::cout << "错误：创建程序失败 " << status << std::endl;
        status = CL_BUILD_PROGRAM_FAILURE;
        return;
    }

    // 7.在线编译程序
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(status != CL_SUCCESS)
    {
        std::cout << "错误：在线编译程序失败 " << status << std::endl;
        return;
    }

    // 8.创建kernel
    kernel = clCreateKernel(program, "kernel_depth2pointcloud", &status);
    if(status != CL_SUCCESS)
    {
        std::cout << "错误：创建kernel失败 " << status << std::endl;
        return;
    }
    // 9.为device（gpu）创建缓冲区
    auto t1 = std::chrono::steady_clock::now();
    float param[7] = {w, h, 520.9, 521.0, 325.1, 249.7, 5000}; // w,h,fx,fy,cx,cy,factor
    cv::Mat grid_cloud = cv::Mat(h, w, CV_32FC3);
    cl_mem mem_rgb = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(uchar)*3*w*h, rgb.data, &status);
    cl_mem mem_depth = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(ushort)*w*h, depth.data, &status);
    cl_mem mem_param = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*5, param, &status);
    cl_mem mem_pointcloud = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*3*w*h, NULL, &status);

    // 10.向kernel传递参数
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_rgb);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_depth);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_param);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_pointcloud);
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "数据加载:" << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / 1000.0 << "ms" << std::endl;

    // 11.运行kernel
    t1 = std::chrono::steady_clock::now();
    size_t localThreads[2] = {32, 4};   //工作组中工作项的排布
    size_t globalThreads[2] = {((w + localThreads[0] - 1) / localThreads[0]) * localThreads[0], 
                               ((h + localThreads[1] - 1) / localThreads[1]) * localThreads[1]};   //整体排布
    // cl_event evt;
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, globalThreads, localThreads, 0, NULL, NULL); //内核执行完成后，会将evt置为CL_SUCCESS/CL_COMPLETE
    // clWaitForEvents(1, &evt);   //等待命令事件发生
    // clReleaseEvent(evt);
    t2 = std::chrono::steady_clock::now();
    std::cout << "GPU运行:" << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / 1000.0 << "ms" << std::endl;

    // 12.从kernel取结果
    t1 = std::chrono::steady_clock::now();
    status = clEnqueueReadBuffer(command_queue, mem_pointcloud, CL_TRUE, 0, sizeof(float)*3*w*h, grid_cloud.data, 0, NULL, NULL);
    t2 = std::chrono::steady_clock::now();
    std::cout << "取数据:" << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() / 1000.0 << "ms" << std::endl;

    clReleaseMemObject(mem_rgb);
    clReleaseMemObject(mem_depth);
    clReleaseMemObject(mem_param);
    clReleaseMemObject(mem_pointcloud);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

/**
 * 深度图转换成点云
*/
void depth2poincloud(const cv::Mat& rgb, const cv::Mat& depth, const Eigen::Matrix3d& intrinsic, const float& depth_factor, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointcloud)
{
    auto t1 = std::chrono::steady_clock::now();

    pointcloud->clear();

    float X, Y, Z;
    for(int v = 0; v < depth.rows; v++)
    {
        for(int u = 0; u < depth.cols; u++)
        {
            // 注意是ushort类型，不是float
            Z = float(depth.at<ushort>(v, u)) / depth_factor;
            if(Z == 0) continue;

            X = (u - intrinsic(0,2)) * Z / intrinsic(0,0);
            Y = (v - intrinsic(1,2)) * Z / intrinsic(1,1);
            cv::Vec3b color = rgb.at<cv::Vec3b>(v, u);
            pcl::PointXYZRGB P;
            P.x = X;
            P.y = Y;
            P.z = Z;
            P.b = color(0);
            P.g = color(1);
            P.r = color(2);
            pointcloud->push_back(P);
        }
    }

    auto t2 = std::chrono::steady_clock::now();
    std::cout << "CPU耗时:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms, point:" << pointcloud->size() << std::endl;

    pcl::io::savePCDFileBinary("./depth.pcd", *pointcloud);
}

void rgb2gray(const cv::Mat& rgb, cv::Mat& gray)
{
    auto t1 = std::chrono::steady_clock::now();
    gray = cv::Mat::zeros(rgb.size(), CV_8UC1);
    for(int v = 0; v < rgb.rows; v++)
    {
        for(int u = 0; u < rgb.cols; u++)
        {
            cv::Vec3b color = rgb.at<cv::Vec3b>(v, u);
            gray.at<uchar>(v, u) = 0.299 * rgb.at<cv::Vec3b>(v, u)(0) + 0.587 * rgb.at<cv::Vec3b>(v, u)(1) + 0.114 * rgb.at<cv::Vec3b>(v, u)(2);
        }
    }
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "cost:" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms, point:"  << std::endl;

}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "usage:./test rgb.png depth.png" << std::endl;
        return -1;
    }

    cv::Mat rgb = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    cv::Mat depth = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    Eigen::Matrix3d intrinsic;
    intrinsic << 520.9, 0, 325.1,
                 0, 521.0, 249.7,
                 0,     0,     1;

    std::cout << "----OpenCL----" << std::endl;
    cl_depth2poincloud(rgb, depth, intrinsic, 5000, pointcloud);

    std::cout << "-----CPU-----" << std::endl;
    depth2poincloud(rgb, depth, intrinsic, 5000, pointcloud);
    return 0;
}