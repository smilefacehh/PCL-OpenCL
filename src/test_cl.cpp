#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

#ifdef APPLE        //平台相关代码
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//编译指令：g++ rgb2gray.cpp `pkg-config --cflags --libs opencv` -lOpenCL -o rgb2gray

void loadProgramSource(const char** files,
                       size_t length,
                       char** buffer,
                       size_t* sizes) {
       /* Read each source file (*.cl) and store the contents into a temporary datastore */
       for(size_t i=0; i < length; i++) {
          FILE* file = fopen(files[i], "r");
          if(file == NULL) {
             perror("Couldn't read the program file");
             exit(1);   
          }
          fseek(file, 0, SEEK_END);
          sizes[i] = ftell(file);
          rewind(file); // reset the file pointer so that 'fread' reads from the front
          buffer[i] = (char*)malloc(sizes[i]+1);
          buffer[i][sizes[i]] = '\0';
          fread(buffer[i], sizeof(char), sizes[i], file);
          fclose(file);
       }
}

int main(void)
{
    Mat srcImage = imread("../data/rgb.png");
    int img_h = srcImage.rows;
    int img_w = srcImage.cols;
    Mat grayImage = Mat(img_h, img_w, CV_8UC1, Scalar(0));  //创建同尺寸灰度图

    cl_int error;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue cQ;

    error = clGetPlatformIDs(1, &platform, NULL);   //获取平台id
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL); //获取设备id
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);    //创建上下文
    cQ = clCreateCommandQueue(context, device, NULL, &error);   //创建命令队列

    const char *file_names[] = {"../kernel/rgb2gray.cl"};     //待编译的内核文件
    const int NUMBER_OF_FILES = 1;
    char* buffer[NUMBER_OF_FILES];
    size_t sizes[NUMBER_OF_FILES];
    loadProgramSource(file_names, NUMBER_OF_FILES, buffer, sizes);  //读取内核文件文本
    program = clCreateProgramWithSource(context, NUMBER_OF_FILES, (const char**)buffer, sizes, &error); //创建program对象
    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);  //编译程序
    if(error != CL_SUCCESS) {
    // If there's an error whilst building the program, dump the log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log = (char*) malloc(log_size+1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
                            log_size+1, program_log, NULL);
        printf("\n=== ERROR ===\n\n%s\n=============\n", program_log);
        free(program_log);
        exit(1);
    }
    error = clCreateKernelsInProgram(program, 1, &kernel, NULL);    //创建内核
    //创建缓存对象
    auto t1 = std::chrono::steady_clock::now();
    cl_mem memRgbImage = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                    sizeof(uchar)*3*img_h*img_w, srcImage.data, &error);    //CL_MEM_COPY_HOST_PTR指定创建缓存对象后拷贝数据
    cl_mem memGrayImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                    sizeof(uchar)*img_h*img_w, NULL, &error);
    cl_mem memImageHeight = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, 
                                    sizeof(int), &img_h, &error);
    cl_mem memImageWidth = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, 
                                    sizeof(int), &img_w, &error);
    //向内核函数传递参数
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memRgbImage);
    error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memGrayImage);
    error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memImageHeight);
    error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memImageWidth);
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "cost1:" << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << std::endl;

    size_t localThreads[2] = {32, 4};   //工作组中工作项的排布
    size_t globalThreads[2] = {((img_w+localThreads[0]-1)/localThreads[0])*localThreads[0], 
                                ((img_h+localThreads[1]-1)/localThreads[1])*localThreads[1]};   //整体排布

    t1 = std::chrono::steady_clock::now();
    cl_event evt;
    error = clEnqueueNDRangeKernel(cQ, kernel,  //启动内核
                               2, 0, globalThreads, localThreads, 
                               0, NULL, &evt);  //内核执行完成后，会将evt置为CL_SUCCESS/CL_COMPLETE
    clWaitForEvents(1, &evt);   //等待命令事件发生
    clReleaseEvent(evt);
    t2 = std::chrono::steady_clock::now();
    std::cout << "cost2:" << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << std::endl;
    //读回数据
    t1 = std::chrono::steady_clock::now();
    error =clEnqueueReadBuffer(cQ, memGrayImage, 
                            CL_TRUE, 
                            0, 
                            sizeof(uchar)*img_h*img_w, 
                            grayImage.data, 0, NULL, NULL);
    t2 = std::chrono::steady_clock::now();
    std::cout << "cost3:" << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << std::endl;

    imshow("srcImage", srcImage);
    imshow("grayImage", grayImage);
    waitKey(0);
    //释放资源
    clReleaseMemObject(memRgbImage);
    clReleaseMemObject(memGrayImage);
    clReleaseMemObject(memImageHeight);
    clReleaseMemObject(memImageWidth);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cQ);
    clReleaseContext(context);

    for(int i = 0; i < NUMBER_OF_FILES; i++)
        free(buffer[i]);
    delete &srcImage;
    delete &grayImage;

    return 0;
}