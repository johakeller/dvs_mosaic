#include <dvs_mosaic/mosaic.h>
#include <dvs_mosaic/image_util.h>
#include <geometry_msgs/PoseStamped.h>
#include <dvs_mosaic/reconstruction.h>
#include <glog/logging.h>

namespace dvs_mosaic
{

  /**
   * \brief Publish several variables related to the mapping (mosaicing) part
   */
  void Mosaic::publishMap()
  {
    // Publish the current map state
    VLOG(1) << "publishMap()";

    if (time_map_pub_.getNumSubscribers() > 0)
    {
      // Time map. Fill content in appropriate range [0,255] and publish
      // Happening at the camera's image plane
      cv_bridge::CvImage cv_image_time;
      cv_image_time.header.stamp = ros::Time::now();
      cv_image_time.encoding = "mono8";
      image_util::normalize(time_map_, cv_image_time.image, 15.);
      time_map_pub_.publish(cv_image_time.toImageMsg());
    }

    // Various mosaic-related topics
    cv_bridge::CvImage cv_image;
    cv_image.header.stamp = ros::Time::now();
    cv_image.encoding = "mono8";
    if (mosaic_pub_.getNumSubscribers() > 0)
    {
      // Brightness image. Fill content in appropriate range [0,255] and publish
      // Call Poisson solver and publish on mosaic_pub_
      // reconstruct from gradient map
      poisson::reconstructBrightnessFromGradientMap(grad_map_, &mosaic_img_);
 //------------------------------------------------------------------------------------------------------------------------------
      // Debug: Check values in mosaic_img_
      //double minVal, maxVal;
      //cv::minMaxLoc(mosaic_img_, &minVal, &maxVal);
      //VLOG(1) << "mosaic_img_ min value: " << minVal << " max value: " << maxVal;
//-----------------------------------------------------------------------------------------------

      // normalize: call image_util::normalize discarding 1% of pixels
      image_util::normalize(mosaic_img_,cv_image.image, 1.0);

      // publish
      mosaic_pub_.publish(cv_image.toImageMsg());
      //VLOG(1) << "_______________________________publishing mosaic___________________________________";
    }

    if (mosaic_gradx_pub_.getNumSubscribers() > 0 ||
        mosaic_grady_pub_.getNumSubscribers() > 0)
    {
      // Visualize gradient map (x and y components)
      cv_bridge::CvImage cv_gradient_x;
      cv_bridge::CvImage cv_gradient_y;
      cv_gradient_x.header.stamp = ros::Time::now();
      cv_gradient_y.header.stamp = ros::Time::now();
      cv_gradient_x.encoding = "mono8";
      cv_gradient_y.encoding = "mono8";

      // use cv::split to split a multi-channel array into its channels (images)
      cv::Mat gradient_components[2];
      // splitting
      cv::split(grad_map_, gradient_components);
      //       call image_util::normalize
      image_util::normalize(gradient_components[0], cv_gradient_x.image, 1.0);
      image_util::normalize(gradient_components[1], cv_gradient_y.image, 1.0);
      // publish
      mosaic_gradx_pub_.publish(cv_gradient_x.toImageMsg());
      mosaic_grady_pub_.publish(cv_gradient_x.toImageMsg());
    }

    if (mosaic_tracecov_pub_.getNumSubscribers() > 0)
    {
      // Visualize confidence: trace of the covariance of the gradient map
      cv_bridge::CvImage cv_covar_trace;
      cv_covar_trace.header.stamp = ros::Time::now();
      cv_covar_trace.encoding = "mono8";
      // coraiance sparse components (3)
      cv::Mat covar_components[3];
      // splitting: use cv::split to split a multi-channel array into its channels (images)
      cv::split(grad_map_covar_, covar_components);
      // tace is sum of 0. component and 2. component
      cv::Mat trace = covar_components[0] + covar_components[1];
      //normalizing: call image_util::normalize
      image_util::normalize(trace, cv_covar_trace.image, 1.0);
      // publish
      mosaic_tracecov_pub_.publish(cv_covar_trace.toImageMsg());
    }
  }

  /**
   * \brief Publish pose once the tracker has estimated it
   */
  void Mosaic::publishPose()
  {
    if (pose_pub_.getNumSubscribers() <= 0)
      return;

    VLOG(1) << "publishPose()";
    geometry_msgs::PoseStamped pose_msg;
    // FILL IN ... when tracking part is implemented

    pose_pub_.publish(pose_msg);
  }

}
