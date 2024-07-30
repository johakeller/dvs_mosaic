#include <dvs_mosaic/mosaic.h>
#include <glog/logging.h>


#include <dvs_mosaic/mosaic.h>
#include <glog/logging.h>
#include <fstream>

namespace dvs_mosaic
{
/*
// CORRECT IMPLEMENTATION
void Mosaic::processEventForMap(const dvs_msgs::Event& ev,
  const double t_ev, const double t_prev,
  const cv::Matx33d& Rot, const cv::Matx33d& Rot_prev)
{
    const int idx = ev.y * sensor_width_ + ev.x;

    // Get time of current and last event at the pixel
    //time_map_.at<double>(ev.y, ev.x) = t_ev;

    const double dt_ev = t_ev - t_prev;
    CHECK_GT(dt_ev, 0) << "Non-positive dt_ev"; // Two events at same pixel with same timestamp


    //const cv::Matx33d Rot;
    //cv::Rodrigues(rot_vec_, Rot); // convert parameter vector to Rotation

    // Get map point corresponding to current event
    // hint: call project_EquirectangularProjection
    cv::Point3d rotated_bvec = Rot * precomputed_bearing_vectors_.at(idx);
    cv::Point2f pm;
    project_EquirectangularProjection(rotated_bvec, pm);

    //VLOG(0) << "pm: [" << pm.x << ", " << pm.y<<"]";

    // Get map point corresponding to previous event at same pixel
    cv::Point3d rotated_bvec_prev = Rot_prev * precomputed_bearing_vectors_.at(idx);
    cv::Point2f pm_prev;
    project_EquirectangularProjection(rotated_bvec_prev, pm_prev);

    // Get approx optical flow vector (vector v in the paper)
    cv::Point2f flow_vec = (pm - pm_prev) / (dt_ev + 1e-9);

    // Extended Kalman Filter (EKF) for the intensity gradient map.
    // Get gradient and covariance at current map point pm
    cv::Vec2f grad_vec = grad_map_.at<cv::Vec2f>(pm);
    cv::Matx21f gm(grad_vec[0], grad_vec[1]);

    cv::Vec3f Pg_vec = grad_map_covar_.at<cv::Vec3f>(pm);
    cv::Matx22f Pg(Pg_vec[0], Pg_vec[1], Pg_vec[1], Pg_vec[2]);


    // Compute innovation, measurement matrix and Kalman gain
    cv::Matx21f dh_dg;
    float nu_innovation;
    if(measure_contrast_)
    {
      // Use contrast as measurement function (Gallego, arXiv 2015)
      dh_dg = cv::Matx21f(flow_vec * dt_ev * (ev.polarity ? 1 : -1));
      nu_innovation = C_th_ - (dh_dg.t() * gm)(0, 0);
    }
    else
    {
      // Use the event rate as measurement function (Kim, BMCV 2014)
      dh_dg = cv::Matx21f(flow_vec / (C_th_ * (ev.polarity ? 1 : -1)));
      nu_innovation = 1.0 / dt_ev - (dh_dg.t() * gm)(0, 0);
    }

    cv::Matx21f Pg_dhdg = Pg * dh_dg;
    const float S_covar_innovation = var_R_ + (dh_dg.t() * Pg_dhdg)(0, 0);
    cv::Matx21f Kalman_gain = (1.f / S_covar_innovation) * Pg_dhdg;

    // Update gradient (state) and covariance
    gm += Kalman_gain * nu_innovation;
    Pg -= Pg_dhdg * Kalman_gain.t();

    // Store updated values of grad_map_ and grad_map_covar_ at corresponding pixel
    grad_map_.at<cv::Vec2f>(pm) = cv::Vec2f(gm(0, 0), gm(1, 0));
    grad_map_covar_.at<cv::Vec3f>(pm) = cv::Vec3f(Pg(0, 0), 0.5f*(Pg(0, 1)+Pg(1, 0)), Pg(1, 1));
  }
*/

/**
* \brief Process each event to refine the mosaic variables (mean and covariance)
*/

// MY IMPLEMENTATION
void Mosaic::processEventForMap(const dvs_msgs::Event& ev,
  const double t_ev, const double t_prev,
  const cv::Matx33d& Rot, const cv::Matx33d& Rot_prev)
{
  const double dt_ev = t_ev - t_prev;
  CHECK_GT(dt_ev,0) << "Non-positive dt_ev"; // Two events at same pixel with same timestamp

  const int index = ev.y * sensor_width_ + ev.x;

  // Get map point corresponding to current event
  // hint: call project_EquirectangularProjection
  cv::Point3d rotated_bvec = Rot * precomputed_bearing_vectors_.at(index);
  cv::Point2f pm;
  project_EquirectangularProjection(rotated_bvec, pm);

  // Get map point corresponding to previous event at same pixel
  cv::Point3d rotated_bvec_prev = Rot_prev * precomputed_bearing_vectors_.at(index);
  cv::Point2f pm_prev;
   project_EquirectangularProjection(rotated_bvec_prev, pm_prev);

  // Get approx optical flow vector (vector v in the paper)
  cv::Point2f flow_vec;
  // avoid zero division
  if (dt_ev != 0.0){
    flow_vec = (pm-pm_prev)/ dt_ev;
  } 
  // if dt_ev zero -> set (0,0)
  else {
    flow_vec = cv::Point2f(0.0, 0.0);
  }

  // Extended Kalman Filter (EKF) for the intensity gradient map.
  // Get gradient and covariance at current map point pm
  // get prev. gradient from gradient map
  cv::Vec2f gradient_curr = grad_map_.at<cv::Vec2f>(pm); 
  // 2x1 matrix
  cv::Matx21f gm(gradient_curr[0], gradient_curr[1]);
  cv::Vec3f pg_vector= grad_map_covar_.at<cv::Vec3f>(pm);  
  cv::Matx22f Pg(pg_vector[0], pg_vector[1],pg_vector[1],pg_vector[2]);


  // Compute innovation, measurement matrix and Kalman gain
  float nu_innovation;

    // get polarity
  int pol;
  if (ev.polarity){
    pol = 1;
  } else {
    pol=-1;
  }
  // calculate innovation nu
  nu_innovation = (1.0/dt_ev)-(gm.dot(flow_vec)/C_th_*pol); 
  
  cv::Matx21f jacobian;
  // calculate Jacobian (delta g/delta h)
  jacobian = cv::Matx21f(flow_vec/C_th_* pol);

  // Pg * Jacobian
  cv::Matx21f Pg_jac = Pg * jacobian;
  // covaraince S
  float covar_s = jacobian.dot(Pg_jac) + var_R_;
  // get Kalman gain
  cv::Matx21f Kalman_gain = Pg_jac *(1.0/covar_s);

  // Update gradient (state) and covariance
  gm += Kalman_gain * nu_innovation;
  Pg -= Kalman_gain * covar_s * Kalman_gain.t();

  // Store updated values on corresponding pixel of grad_map_ and grad_map_covar_
    grad_map_.at<cv::Vec2f>(pm)= cv::Vec2f(gm(0, 0), gm(1, 0));

    // 3D covariance vector
    cv::Vec3f covar_vector(Pg(0,0), Pg(0,1), Pg(1,1));
    grad_map_covar_.at<cv::Vec3f>(pm) = covar_vector;
}

}
