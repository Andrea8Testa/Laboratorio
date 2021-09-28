// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_impedance_MBRL_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka_example_controllers/pseudo_inversion.h>

#include <fstream>
#include <unistd.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/WrenchStamped.h>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>
//#include "examples_common.h"
#include <iostream>

#include <nlopt.h>
#include <vector>
#include <utility>
#include <math.h>
#include <time.h>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <array>
#include <functional>

namespace franka_example_controllers {

    bool CartesianImpedanceMBRLController::init(hardware_interface::RobotHW* robot_hw,
        ros::NodeHandle& node_handle) {
        std::vector<double> cartesian_stiffness_vector;
        std::vector<double> cartesian_damping_vector;
        // Topics
        franka_EE_pose_pub = node_handle.advertise<geometry_msgs::PoseStamped>("/franka_ee_pose", 20);
        franka_EE_velocity_pub = node_handle.advertise<geometry_msgs::TwistStamped>("/franka_ee_velocity", 20);
        franka_EE_wrench_pub = node_handle.advertise<geometry_msgs::WrenchStamped>("/franka_ee_wrench", 20); // 1000
        franka_q_velocity_pub = node_handle.advertise<geometry_msgs::PoseStamped>("/franka_q_velocity", 20);
        franka_q_pose_pub = node_handle.advertise<geometry_msgs::PoseStamped>("/franka_q_pose", 20); // 1000

        sub_equilibrium_pose_ = node_handle.subscribe(
            "/QLMPC_pose", 20, &CartesianImpedanceMBRLController::equilibriumPoseCallback, this,
            ros::TransportHints().reliable().tcpNoDelay());

        sub_damping_ = node_handle.subscribe("/D_information", 20, &CartesianImpedanceMBRLController::dampingCallback, this,
            ros::TransportHints().reliable().tcpNoDelay()); //20 previous queue
        sub_stiffness_ = node_handle.subscribe("/K_information", 20, &CartesianImpedanceMBRLController::stiffnessCallback, this,
            ros::TransportHints().reliable().tcpNoDelay()); //20 previous queue

        std::string arm_id;
        if (!node_handle.getParam("arm_id", arm_id)) {
            ROS_ERROR_STREAM("CartesianImpedanceMBRLController: Could not read parameter arm_id");
            return false;
        }
        std::vector<std::string> joint_names;
        if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
            ROS_ERROR(
                "CartesianImpedanceMBRLController: Invalid or no joint_names parameters provided, "
                "aborting controller init!");
            return false;
        }

        auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
        if (model_interface == nullptr) {
            ROS_ERROR_STREAM(
                "CartesianImpedanceMBRLController: Error getting model interface from hardware");
            return false;
        }
        try {
            model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
                model_interface->getHandle(arm_id + "_model"));
        }
        catch (hardware_interface::HardwareInterfaceException& ex) {
            ROS_ERROR_STREAM(
                "CartesianImpedanceMBRLController: Exception getting model handle from interface: "
                << ex.what());
            return false;
        }

        auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
        if (state_interface == nullptr) {
            ROS_ERROR_STREAM(
                "CartesianImpedanceMBRLController: Error getting state interface from hardware");
            return false;
        }
        try {
            state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
                state_interface->getHandle(arm_id + "_robot"));
        }
        catch (hardware_interface::HardwareInterfaceException& ex) {
            ROS_ERROR_STREAM(
                "CartesianImpedanceMBRLController: Exception getting state handle from interface: "
                << ex.what());
            return false;
        }

        auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
        if (effort_joint_interface == nullptr) {
            ROS_ERROR_STREAM(
                "CartesianImpedanceMBRLController: Error getting effort joint interface from hardware");
            return false;
        }
        for (size_t i = 0; i < 7; ++i) {
            try {
                joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
            }
            catch (const hardware_interface::HardwareInterfaceException& ex) {
                ROS_ERROR_STREAM(
                    "CartesianImpedanceMBRLController: Exception getting joint handles: " << ex.what());
                return false;
            }
        }

        position_d_.setZero();
        orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
        position_d_target_.setZero();
        orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

        cartesian_stiffness_.setZero();
        cartesian_damping_.setZero();

        return true;
    }

    void CartesianImpedanceMBRLController::starting(const ros::Time& /*time*/) {
        // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
        // to initial configuration
        initial_state = state_handle_->getRobotState();

        Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
        Eigen::Vector3d position_init_temp(initial_transform.translation());
        position_init = position_init_temp;

        // set equilibrium point to current state
        position_d_ = initial_transform.translation();
        orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
        position_d_target_ = initial_transform.translation();
        orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());

        Eigen::Map<const Eigen::Matrix<double, 7, 1> > joint_pos_d_init(initial_state.q.data());
        joint_velocity_d.setZero();
        joint_pos_d.setZero();
        int_jnt_pos_err.setZero();
        dq_filt_Eigen.setZero();
        for (int j = 0; j < 7; ++j)
            joint_pos_d(j) = joint_pos_d_init(j,0);


        cont_task_setpoint = 0;
        initialization = false;

        rpy_init.setZero();
        vel_imp_t.setZero();
        pos_imp_t.setZero();
        vel_imp_r.setZero();
        pos_imp_r.setZero();
        w_psp_world.setZero();

        Kp.setZero();
        Kd.setZero();
        Ki.setZero();

        Kpos.setZero();
        for (int j = 0; j < 6; ++j)
            Kpos(j,j) = 0.25;

        Kp(0,0) = 500.; //1000.
        Kp(1,1) = 750.; //1500.
        Kp(2,2) = 750.; //1500.
        Kp(3,3) = 750.; //1500.
        Kp(4,4) = 400.;
        Kp(5,5) = 200.;
        Kp(6,6) = 150.;

        Kd(0,0) = 16.; //32.
        Kd(1,1) = 20.; //40.
        Kd(2,2) = 15.; //30.
        Kd(3,3) = 22.5; //45.
        Kd(4,4) = 10.; //20.
        Kd(5,5) = 10.; //20.
        Kd(6,6) = 5.; //10. rimasto 10 nella prova

        Ki(0,0) = 7.5; //15.
        Ki(1,1) = 7.5; //15.
        Ki(2,2) = 10.; //20.
        Ki(3,3) = 20.; //40.
        Ki(4,4) = 20.; //40.
        Ki(5,5) = 15.; //30. rimasto 30 nella prova
        Ki(6,6) = 20.; //40. rimasto 40 nelòla prova


        for (int j = 0; j < 3; ++j)
        {
            dw_psp_world(j) = 0.;
            drpy_cmd(j) = 0.;
            rpy_cmd(j) = 0.;
            drpy(j) = 0.;
            rpy_old(j) = 0.;
            dposition(j) = 0.;
            position_old(j) = 0.;
            dposition_filt(j) = 0.;
            drpy_filt(j) = 0.;
        }

        for (int j = 0; j < 7; ++j)
            dq_filt[j] = 0.;

        // impedance parameters definition
        h_damp_t = 0.7; //1.;                  // last "Roveda" value = 0.75
        h_damp_r = 0.7; //1.;                  // last "Roveda" value = 0.75
        damping_importato = h_damp_t;
        damping_old = h_damp_t;
        mass_imp = 10.; //5.
        inertia_imp = 10.; //10.
        translational_stiffness = 500.;//1000.; // last "Roveda" value = 7500
        rotational_stiffness = 50.;//50.;   // last "Roveda" value = 15000
        stiffness_importata = translational_stiffness;

        msrTimestep = 0.001;
        filtTime = msrTimestep * 2.;
        digfilt = 0.;

        if (filtTime > 0.)
            digfilt = exp(-msrTimestep / filtTime); // 0.6
    }

    void CartesianImpedanceMBRLController::update(const ros::Time& /*time*/,
        const ros::Duration& /*period*/) {
        // get state variables
        std::array<double, 49> inertia_array = model_handle_->getMass();
        franka::RobotState robot_state = state_handle_->getRobotState();
        std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
        std::array<double, 42> jacobian_array =
            model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

        // convert to Eigen
        Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
        Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 7> > inertia(inertia_array.data());
        Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
        Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
        Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(  // NOLINT (readability-identifier-naming)
            robot_state.tau_J_d.data()); // Desired link-side joint torque sensor signals without gravity
        Eigen::Map<const Eigen::Matrix<double, 7, 1> > tau_ext(robot_state.tau_ext_hat_filtered.data());
        // load from the robot the updated state
        Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
        Eigen::Vector3d position(transform.translation());
        Eigen::Quaterniond orientation(transform.linear());
        // load from the robot the updated wrench
        Eigen::Map<Eigen::Matrix<double, 6, 1>> wrench_v(robot_state.O_F_ext_hat_K.data());
        // impedance matrixes definition
        Eigen::MatrixXd stiffness(6, 6), damping(6, 6), mass(6, 6), inv_mass(6, 6);
        mass.setZero();
        mass.topLeftCorner(3, 3) << mass_imp * Eigen::MatrixXd::Identity(3, 3);
        mass.bottomRightCorner(3, 3) << inertia_imp * Eigen::MatrixXd::Identity(3, 3);
        stiffness.setZero();
        stiffness.topLeftCorner(3, 3) << translational_stiffness * Eigen::MatrixXd::Identity(3, 3);
        stiffness.bottomRightCorner(3, 3) << rotational_stiffness * Eigen::MatrixXd::Identity(3, 3);
        damping.setZero();
        damping.topLeftCorner(3, 3) << 2.0 * mass_imp * h_damp_t * sqrt(translational_stiffness / mass_imp) *
            Eigen::MatrixXd::Identity(3, 3);
        damping.bottomRightCorner(3, 3) << 2.0 * inertia_imp * h_damp_r * sqrt(rotational_stiffness / inertia_imp) *
            Eigen::MatrixXd::Identity(3, 3);

        if (!initialization){
        // convert to eigen
        Eigen::Vector3d position_init_temp(transform.translation());
        position_init = position_init_temp;

        // set equilibrium point to current state
        position_d_ = transform.translation();
        orientation_d_ = Eigen::Quaterniond(transform.linear());
        position_d_target_ = transform.translation();
        orientation_d_target_ = Eigen::Quaterniond(transform.linear());

        initialization = true;
        }

        // q publisher
        geometry_msgs::PoseStamped q_msg;
        q_msg.pose.position.x = q[0];
        q_msg.pose.position.y = q[1];
        q_msg.pose.position.z = q[2];
        q_msg.pose.orientation.x = q[3];
        q_msg.pose.orientation.y = q[4];
        q_msg.pose.orientation.z = q[5];
        q_msg.pose.orientation.w = q[6];

        franka_q_pose_pub.publish(q_msg);

        // dq publisher
        geometry_msgs::PoseStamped dq_msg;
        dq_msg.pose.position.x = dq[0];
        dq_msg.pose.position.y = dq[1];
        dq_msg.pose.position.z = dq[2];
        dq_msg.pose.orientation.x = dq[3];
        dq_msg.pose.orientation.y = dq[4];
        dq_msg.pose.orientation.z = dq[5];
        dq_msg.pose.orientation.w = dq[6];

        franka_q_velocity_pub.publish(dq_msg);

        // // saturation set point
        // if ((position_d_target_[2] - position_d_[2]) > 0.05) {
        //     position_d_target_[2] = position_d_[2] + 0.05;
        // }
        // else if ((position_d_target_[2] - position_d_[2]) < -0.05) {
        //     position_d_target_[2] = position_d_[2] - 0.05;
        // }

        // // saturation damping
        // if ((damping_importato - damping_old) > 0.1) {
        //     damping_importato = damping_old + 0.1;
        // }
        // else if ((damping_importato - damping_old) < -0.1) {
        //     damping_importato = damping_old - 0.1;
        // }

        damping(3, 3) = damping_importato * 2.0 * mass_imp * sqrt(translational_stiffness / mass_imp);
        stiffness(3, 3) = stiffness_importata;

        // pose publisher
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.pose.position.x = position(0);
        pose_msg.pose.position.y = position(1);
        pose_msg.pose.position.z = position(2);
        pose_msg.pose.orientation.x = orientation.x();
        pose_msg.pose.orientation.y = orientation.y();
        pose_msg.pose.orientation.z = orientation.z();
        pose_msg.pose.orientation.w = orientation.w();

        franka_EE_pose_pub.publish(pose_msg);

        orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);

        cont_task_setpoint++;

        Eigen::Affine3d transform_cmd(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
        Eigen::Matrix3d R_msr(transform.rotation());
        Eigen::Matrix3d R_cmd(transform_cmd.rotation());
        Eigen::Matrix3d R_comp = R_msr.inverse() * R_cmd;

        Eigen::Matrix3d T_dangles_to_w_cd;
        Eigen::Matrix3d dT_dangles_to_w_cd;
        Eigen::Vector3d rpy; //euler angles in RPY convention

        rpy(2) = atan2(R_comp(1,0)  ,R_comp(0,0));
        rpy(1) = atan2(-R_comp(2,0) ,pow( ( pow(R_comp(2,1), 2) + pow( R_comp(2,2), 2) ), 0.5 ) );
        rpy(0) = atan2(R_comp(2,1)  ,R_comp(2,2));

        for (int j = 0; j < 7; ++j)
            dq_filt[j] = dq_filt[j] * digfilt + (1 - digfilt) * dq(j);
        // cartesian velocity calculation
        for (int j = 0; j < 3; ++j)
        {
            dposition(j) = (position(j) - position_old(j)) / 0.001;
            dposition_filt(j) = dposition_filt(j) * digfilt + (1 - digfilt) * dposition(j);
            drpy(j) = (rpy(j) - rpy_old(j)) / 0.001;
            drpy_filt(j) = drpy_filt(j) * digfilt + (1 - digfilt) * drpy(j);
        }
        // velocity publisher
        geometry_msgs::TwistStamped velocity_msg;
        velocity_msg.twist.linear.x = dposition_filt(0);
        velocity_msg.twist.linear.y = dposition_filt(1);
        velocity_msg.twist.linear.z = dposition_filt(2);
        velocity_msg.twist.angular.x = drpy_filt(0);
        velocity_msg.twist.angular.y = drpy_filt(1);
        velocity_msg.twist.angular.z = drpy_filt(2);

        franka_EE_velocity_pub.publish(velocity_msg);

        Eigen::Vector3d ext_wrench_t, ext_wrench_r;
        Eigen::VectorXd ext_wrench(6);
        Eigen::MatrixXd Jpinv(7,6);
        Eigen::MatrixXd JpinvT(6,7);

        Jpinv = jacobian.completeOrthogonalDecomposition().pseudoInverse();
        JpinvT = (jacobian.transpose()).completeOrthogonalDecomposition().pseudoInverse();
        double th_F_msr = 2.;
        ext_wrench = -(JpinvT*tau_ext).transpose();

        for (int j = 0; j < 3; ++j){
            ext_wrench_t(j) = ext_wrench(j);
            ext_wrench_r(j) = ext_wrench(j+3);

        }

        if (fabs(ext_wrench_t(2))>th_F_msr)
        {
            if (ext_wrench_t(2)>0.)
                ext_wrench_t(2) = ext_wrench_t(2) - th_F_msr;
            else
                ext_wrench_t(2) = ext_wrench_t(2) + th_F_msr;
        }
        else
            ext_wrench_t(2) = 0.;


        // wrench publisher
        geometry_msgs::WrenchStamped wrench_msg;
        wrench_msg.wrench.force.x = ext_wrench_t(0);
        wrench_msg.wrench.force.y = ext_wrench_t(1);
        wrench_msg.wrench.force.z = ext_wrench_t(2);
        wrench_msg.wrench.torque.x = ext_wrench_r(0);
        wrench_msg.wrench.torque.y = ext_wrench_r(1);
        wrench_msg.wrench.torque.z = ext_wrench_r(2);

        franka_EE_wrench_pub.publish(wrench_msg);

        position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;

        // if (cont_task_setpoint < 5000)
        //     position_d_(2) = pos_imp_t(2) + position_init(2) + 0.001 * cos(5*cont_task_setpoint/1000);

        // else if (cont_task_setpoint == 5000)
        //        position_d_(2) = pos_imp_t(2) + position_init(2) + 0.001 * cos(5*5);
        position_d_(2) = pos_imp_t(2) + position_init(2) - 0.1 * ext_wrench_t(2)/stiffness(2,2);
        if (position_d_(2) > 0.65)
                position_d_(2) = 0.65;
        if (position_d_(2) < 0.35)
                position_d_(2) = 0.35;

        inv_mass = mass.inverse();

        position_old = position;
        rpy_old = rpy;
        damping_old = damping_importato;

        Eigen::Vector3d acc_imp_t, acc_imp_r;

        for (int j = 0; j < 3; ++j)
        {
            acc_imp_t(j) = 0.;
            acc_imp_r(j) = 0.;
        }

        T_dangles_to_w_cd(0,0) = 1;
        T_dangles_to_w_cd(0,1) = 0.;
        T_dangles_to_w_cd(0,2) = sin(rpy(1));
        T_dangles_to_w_cd(1,0) = 0.;
        T_dangles_to_w_cd(1,1) = cos(rpy(0));
        T_dangles_to_w_cd(1,2) = -sin(rpy(0))*cos(rpy(1));
        T_dangles_to_w_cd(2,0) = 0.;
        T_dangles_to_w_cd(2,1) = sin(rpy(0));
        T_dangles_to_w_cd(2,2) = cos(rpy(0))*cos(rpy(1));

        dT_dangles_to_w_cd(0,0) = 0.;
        dT_dangles_to_w_cd(0,1) = 0.;
        dT_dangles_to_w_cd(0,2) = cos(rpy(1))*drpy_filt(1);
        dT_dangles_to_w_cd(1,0) = 0.;
        dT_dangles_to_w_cd(1,1) = -sin(rpy(0))*drpy_filt(0);
        dT_dangles_to_w_cd(1,2) = -cos(rpy(0))*cos(rpy(1))*drpy_filt(0) + sin(rpy(0))*sin(rpy(1))*drpy_filt(1);
        dT_dangles_to_w_cd(2,0) = 0.;
        dT_dangles_to_w_cd(2,1) = cos(rpy(0))*drpy_filt(0);
        dT_dangles_to_w_cd(2,2) = -( sin(rpy(0))*cos(rpy(1))*drpy_filt(0) + cos(rpy(0))*sin(rpy(1))*drpy_filt(1) );

        acc_imp_t = inv_mass.topLeftCorner(3,3) * ( ext_wrench_t - stiffness.topLeftCorner(3,3) * (pos_imp_t + position_init - position_d_) - damping.topLeftCorner(3,3) * vel_imp_t );
        acc_imp_r = inv_mass.bottomRightCorner(3,3) * ( T_dangles_to_w_cd.transpose()*(R_msr.inverse()*ext_wrench_r) + stiffness.bottomRightCorner(3,3) * rpy + damping.bottomRightCorner(3,3) * drpy_filt );

        Eigen::VectorXd cmd_cart_pose(6), impdamp_t(3), impstiff_t(3);

        for (int j = 0; j < 3; ++j)
        {
            vel_imp_t(j) += acc_imp_t(j)*0.001;
            pos_imp_t(j) += vel_imp_t(j)*0.001;
            vel_imp_r(j) += acc_imp_r(j)*0.001;
            pos_imp_r(j) += vel_imp_r(j)*0.001;
            cmd_cart_pose(j) = pos_imp_t(j) + position_init(j);
            cmd_cart_pose(j+3) = pos_imp_r(j);
            impdamp_t(j) = damping(j,j);
            impstiff_t(j) = stiffness(j,j);
        }

        // for (int j = 0; j < 3; ++j)
        // {
        //     drpy_cmd(j) = acc_imp_r(j)*0.001;
        //     rpy_cmd(j) = drpy_cmd(j)*0.001;
        // }

        Eigen::Matrix3d T_cd_world = R_msr*T_dangles_to_w_cd;
        Eigen::Matrix3d dT_cd_world = R_msr*dT_dangles_to_w_cd;

        dw_psp_world = T_cd_world*acc_imp_r + dT_cd_world*drpy_filt;
        w_psp_world += dw_psp_world*0.001;

        // Eigen::Matrix<double, 6, 1> acc_cmd;
        // acc_cmd(0) = acc_imp_t(0);
        // acc_cmd(1) = acc_imp_t(1);
        // acc_cmd(2) = acc_imp_t(2);
        // acc_cmd.tail(3) = dw_psp_world;

        Eigen::VectorXd velocity_d_(6);
        velocity_d_.setZero();
        velocity_d_(0) = vel_imp_t(0);
        velocity_d_(1) = vel_imp_t(1);
        velocity_d_(2) = vel_imp_t(2);
        velocity_d_.tail(3) = w_psp_world;

        ///////////////////Old control///////////////////

        // Eigen::MatrixXd Jpinv(7,6);
        // Jpinv = jacobian.completeOrthogonalDecomposition().pseudoInverse();

        // // compute control
        // Eigen::VectorXd tau_imp(7), tau_d(7), tau_nullspace(7);

        // Eigen::MatrixXd NullSpace(7, 7);

        // for (int j = 0; j < 7; ++j)
        //     tau_nullspace(j) = 0.;

        // NullSpace = Eigen::MatrixXd::Identity(7, 7) - Jpinv * jacobian;

        // Eigen::MatrixXd M_nullspace(7, 7), D_nullspace(7, 7);

        // double mns = 10.;
        // double kns = 1000.; // last "Roveda" value = 1000
        // double hns = 1.5; //5
        // double dns = 2 * hns * mns * pow((kns / mns), 0.5);

        // M_nullspace = mns * Eigen::MatrixXd::Identity(7, 7);
        // D_nullspace = dns * Eigen::MatrixXd::Identity(7, 7);

        // Eigen::VectorXd dq_filt_Eigen(7);
        // for (int j = 0; j < 7; ++j)
        //     dq_filt_Eigen(j) = dq_filt[j];
        // tau_nullspace = inertia * (NullSpace * (-M_nullspace.inverse() * D_nullspace * dq_filt_Eigen));

        // tau_imp << inertia * (Jpinv * acc_cmd); // once we get the required acceleration, we can obtain the torques
        // tau_d << tau_imp + coriolis + tau_nullspace; // + (jacobian.transpose()*ext_wrench);

        ///////////////////Position control///////////////////

        for (int j = 0; j < 7; ++j)
                dq_filt_Eigen(j) = dq_filt_Eigen(j) * digfilt + (1-digfilt) * dq(j);

        // compute control
        Eigen::VectorXd tau_ctrl(7), tau_d(7), ctrl_velocity(6), position_err(6);

        ctrl_velocity.setZero();
        position_err.setZero();
        position_err(0) = pos_imp_t(0) + position_init(0) - position(0);
        position_err(1) = pos_imp_t(1) + position_init(1) - position(1);
        position_err(2) = pos_imp_t(2) + position_init(2) - position(2);

        ctrl_velocity = velocity_d_ + Kpos*position_err;

        joint_velocity_d.setZero();
        joint_velocity_d = Jpinv*ctrl_velocity;

        joint_pos_d += joint_velocity_d*0.001;

        int_jnt_pos_err += (joint_pos_d - q)*0.001;

        tau_ctrl << Kp*(joint_pos_d - q) + Kd*(joint_velocity_d - dq_filt_Eigen) + Ki*int_jnt_pos_err; // + Ki*int_jnt_pos_err;

        tau_d << tau_ctrl + coriolis;

        Eigen::VectorXd tau_friction(7), sign_dq(7), mask_dq(7);

        std::array<double, 7> max_f1 = {1.5,1.5,1.5,1.5,1.5,1.5,1.5};
        std::array<double, 7> max_f2 = {1.5,1.5,1.5,1.5,1.5,1.5,1.5};
        std::array<double, 7> max_f3 = {1.5,1.5,1.5,1.5,1.5,1.5,1.5};
        std::array<double, 7> max_f4 = {10.,10.,10.,10.,10.,10.,10.};
        std::array<double, 7> max_f5 = {2.5,2.5,2.5,2.5,2.5,2.5,2.5};

        std::array<double, 7> min_f1 = {0.,0.,0.,0.,0.,0.,0.};
        std::array<double, 7> min_f2 = {0.,0.,0.,0.,0.,0.,0.};
        std::array<double, 7> min_f3 = {0.,0.,0.,0.,0.,0.,0.};
        std::array<double, 7> min_f4 = {0.,0.,0.,0.,0.,0.,0.};
        std::array<double, 7> min_f5 = {0.,0.,0.,0.,0.,0.,0.};

        std::array<double, 35> friction_param = {{ 0.5, 0.67284, 0.722222, 0.228395, 0.833333, 0.709877, 0.796296, 0.685185, 0.771605, 0.648148, 0.685185, 0.277778, 0.154321, 0.462963, 0.240741, 0.32716, 0.907407, 0.87037, 0.117284, 0.314815, 0.648148, 0.314815, 0.277778, 0.907407, 0.759259, 0.796296, 0.685185, 0.685185, 0.660494, 0.524691, 0.166667, 0.685185, 0.438272, 0.722222, 0.314815 }};

        for (int j = 0; j < 7; ++j)
        {

            if (abs(tau_d(j))>0.01)
            {
                mask_dq(j) = 1.;
                sign_dq(j) = tau_d(j)/abs(tau_d(j));
            }
            else
            {
                mask_dq(j) = 0.;
                sign_dq(j) = 0.;
            }

            double p1 = (min_f1[j] + friction_param[j]/5.*(max_f1[j]-min_f1[j]));
            double p2 = (min_f2[j] + friction_param[j+7]/5.*(max_f2[j]-min_f2[j]));
            double p3 = (min_f3[j] + friction_param[j+14]/5.*(max_f3[j]-min_f3[j]));
            double p4 = (min_f4[j] + friction_param[j+21]/5.*(max_f4[j]-min_f4[j]));
            double p5 = (min_f5[j] + friction_param[j+28]/5.*(max_f5[j]-min_f5[j]));

            tau_friction(j) = 0.;

            tau_friction(j) = - ( p1*dq_filt_Eigen(j)*mask_dq(j) - (p2 + (p3)*exp(-pow( abs( dq_filt_Eigen(j)*p4 ) , p5)))*sign_dq(j) );
        }

        tau_d += tau_friction;

        // to print in the first 10 instants
        if (cont_task_setpoint % 1 == 0 && cont_task_setpoint < 10)
        {
        // print

        // printf("external wrench t z \n");
        // printf("%f \n", ext_wrench_t(2));

        // printf("position init \n");
        // printf("%f \n",  position_init(2) );

        // printf("pos_imp_t \n");
        // printf("%f \n",  pos_imp_t(2) );

        // printf("position_d_ \n");
        // printf("%f \n",  position_d_(2) );

        // printf("acc_imp_t \n");
        // printf("%f \n",  acc_imp_t(2) );

        // for (int j = 0; j < 7; ++j){
        // printf("tau_d\n");
        // printf("%f \n",tau_d(j));
        // }
        // for (int j = 0; j < 7; ++j){
        // printf("tau_friction\n");
        // printf("%f \n",tau_friction(j));
        // }
        // for (int j = 0; j < 7; ++j){
        // printf("tau_ctrl\n");
        // printf("%f \n",tau_ctrl(j));
        // }
        // printf("damping_importato \n");
        // printf("%f \n", damping_importato );

        // for (int j = 0; j < 7; ++j){
        // printf("joint_pos_d - q\n");
        // printf("%f \n",joint_pos_d(j) - q(j));
        // }

        //printf("********************** \n");

        }

        // to print in all the instants

        printf("external wrench t z \n");
        printf("%f \n", ext_wrench_t(2));

        printf("position_d_ \n");
        printf("%f \n",  position_d_(2) );

        printf("position_d_target_ \n");
        printf("%f \n",  position_d_target_(2) );

        printf("K \n");
        printf("%f \n", stiffness(3,3) );

        // printf("K2 \n");
        // printf("%f \n", stiffness(2,2) );

        printf("D \n");
        printf("%f \n", damping(3,3) );

        // printf("D2 \n");
        // printf("%f \n", damping(2,2) );

        // printf("position init \n");
        // printf("%f \n",  position_init(2) );

        // printf("pos_imp_t \n");
        // printf("%f \n",  pos_imp_t(2) );

        printf("ext_wrench \n");
        printf("%f \n", ext_wrench(2) );

        printf("********************** \n");


        std::array<double, 7> tau_d_array{};
        Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

        // Saturate torque rate to avoid discontinuities
        tau_d << saturateTorqueRate(tau_d, tau_J_d);
        for (size_t i = 0; i < 7; ++i) {
            joint_handles_[i].setCommand(tau_d(i));
        }

        /*
        // update parameters changed online either through dynamic reconfigure or through the interactive target by filtering
        cartesian_damping_ = filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
        */
    }

    Eigen::Matrix<double, 7, 1> CartesianImpedanceMBRLController::saturateTorqueRate(
        const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
        const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
        Eigen::Matrix<double, 7, 1> tau_d_saturated{};
        for (size_t i = 0; i < 7; i++) {
            double difference = tau_d_calculated[i] - tau_J_d[i];
            tau_d_saturated[i] =
                tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_); //20 in header
        }
        return tau_d_saturated;
    }

    void CartesianImpedanceMBRLController::equilibriumPoseCallback(
        const geometry_msgs::PoseStampedConstPtr& msg) {
        position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
        Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
        orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
            msg->pose.orientation.z, msg->pose.orientation.w;
        if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
            orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
        }
    }

    void CartesianImpedanceMBRLController::dampingCallback(const std_msgs::Float64::ConstPtr& msg) {
        damping_importato = msg->data;
    }

    void CartesianImpedanceMBRLController::stiffnessCallback(const std_msgs::Float64::ConstPtr& msg) {
        stiffness_importata = msg->data;
    }

}

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceMBRLController,
    controller_interface::ControllerBase)
