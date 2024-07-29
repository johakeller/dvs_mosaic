#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

namespace image_util
{

void minMaxLocRobust(const cv::Mat& image, float& rmin, float& rmax,
                     const float& percentage_pixels_to_discard);

void normalize(const cv::Mat& src, cv::Mat& dst, const float& percentage_pixels_to_discard);

} // namespace
