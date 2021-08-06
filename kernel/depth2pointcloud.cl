__kernel void kernel_depth2pointcloud(__global uchar* rgb, 
                                      __global ushort* depth, 
                                      __global const float* param, 
                                      __global float* grid_pointcloud)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    float w = param[0], h = param[1], fx = param[2], fy = param[3], cx = param[4], cy = param[5], factor = param[6];

    if(x < w && y < h)
    {
        int index = y * w + x;
        float Z = depth[index] / factor;
        float X = 0, Y = 0;
        if(Z > 0)
        {
            X = (x - cx) * Z / fx;
            Y = (y - cy) * Z / fy;
        }
        grid_pointcloud[3*index] = X;
        grid_pointcloud[3*index+1] = Y;
        grid_pointcloud[3*index+2] = Z;
    }
}