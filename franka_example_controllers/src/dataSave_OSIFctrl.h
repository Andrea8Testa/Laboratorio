
#include <iostream>
#include <cmath>
#include <sys/types.h> 
#include <unistd.h>
#include <sched.h>
#include <errno.h>  
#include <time.h>
#include <signal.h>
#include <string>
#include <algorithm>

namespace datasaving_OSIFctrl{

	class SaveData_OSIFctrl{

public:

SaveData_OSIFctrl(std::string usrdef_savepath,
	 std::string usrdef_fileext,
	 bool usrdef_bool_max_size,
	 int usrdef_max_size);

void WriteData2File(std::array<double, 16> cart_pose, Eigen::VectorXd cmd_imp_cart_pose_t, Eigen::VectorXd cmd_cart_pose, Eigen::VectorXd cmd_imp_cart_vel_t, double impact_vel_d, double alpha_fuzzy, Eigen::VectorXd torquesmsr, Eigen::VectorXd torquesEKF, Eigen::VectorXd wrenchmsr, Eigen::VectorXd wrenchEKF, Eigen::VectorXd dqcmd, Eigen::VectorXd qcmd, Eigen::VectorXd qmsr, Eigen::VectorXd imp_damping_t, Eigen::VectorXd imp_stiffness_t, double time);

void ClosingDataFiles();

private:
	int elements_count;
	std::string savepath;
	std::string fileext;
    	bool max_size_files_on;
    	int max_size_file;
	
    FILE * ptr_file_cart_pose; std::string savefile_cart_pose;
    
    FILE * ptr_file_cmd_imp_cart_pose_t; std::string savefile_cmd_imp_cart_pose_t;
    
    FILE * ptr_file_cmd_cart_pose; std::string savefile_cmd_cart_pose;
    
    FILE * ptr_file_cmd_imp_cart_vel_t; std::string savefile_cmd_imp_cart_vel_t;
    
    FILE * ptr_file_impactveld; std::string savefile_impactveld;
    
    FILE * ptr_file_alpha_fuzzy; std::string savefile_alpha_fuzzy;
    
    FILE * ptr_file_qmsr; std::string savefile_qmsr;
    
    FILE * ptr_file_dqcmd; std::string savefile_dqcmd;
    
    FILE * ptr_file_qcmd; std::string savefile_qcmd;
    
    FILE * ptr_file_wrenchmsr; std::string savefile_wrenchmsr;
    
    FILE * ptr_file_wrenchEKF; std::string savefile_wrenchEKF;
    
    FILE * ptr_file_torquesmsr; std::string savefile_torquesmsr;
    
    FILE * ptr_file_torquesEKF; std::string savefile_torquesEKF;
    
    FILE * ptr_file_impdamp_t; std::string savefile_impdamp_t;
    
    FILE * ptr_file_impstiff_t; std::string savefile_impstiff_t;
};

inline SaveData_OSIFctrl::SaveData_OSIFctrl(std::string usrdef_savepath,
			  std::string usrdef_fileext,
			  bool usrdef_bool_max_size,
			  int usrdef_max_size){

    savepath = usrdef_savepath; //"/tmp/";
    fileext = usrdef_fileext;
    max_size_files_on = usrdef_bool_max_size;
    max_size_file = usrdef_max_size;
    elements_count = 1;
    
    savefile_cart_pose = savepath + "cart_pose_" + fileext + ".txt";
    
    savefile_cmd_imp_cart_pose_t = savepath + "cmd_imp_cart_pose_t_" + fileext + ".txt";
    
    savefile_cmd_cart_pose = savepath + "cmd_cart_pose_" + fileext + ".txt";
    
    savefile_cmd_imp_cart_vel_t = savepath + "cmd_imp_cart_vel_t_" + fileext + ".txt";
    
    savefile_impactveld = savepath + "impactveld_" + fileext + ".txt";
    
    savefile_alpha_fuzzy = savepath + "alpha_fuzzy_" + fileext + ".txt";
    
    savefile_dqcmd = savepath + "dqcmd_" + fileext + ".txt";
    
    savefile_qcmd = savepath + "qcmd_" + fileext + ".txt";
    
    savefile_qmsr = savepath + "qmsr_" + fileext + ".txt";
    
    savefile_wrenchmsr = savepath + "wrenchmsr_" + fileext + ".txt";
    
    savefile_wrenchEKF = savepath + "wrenchEKF_" + fileext + ".txt";
    
    savefile_torquesmsr = savepath + "torquesmsr_" + fileext + ".txt";
    
    savefile_torquesEKF = savepath + "torquesEKF_" + fileext + ".txt";
    
    savefile_impdamp_t = savepath + "impdamp_t_" + fileext + ".txt";
    
    savefile_impstiff_t = savepath + "impstiff_t_" + fileext + ".txt";
        
    ptr_file_cart_pose = fopen ( savefile_cart_pose.c_str() , "w" );
    
    ptr_file_cmd_imp_cart_pose_t = fopen ( savefile_cmd_imp_cart_pose_t.c_str() , "w" );
    
    ptr_file_cmd_cart_pose = fopen ( savefile_cmd_cart_pose.c_str() , "w" );
    
    ptr_file_cmd_imp_cart_vel_t = fopen ( savefile_cmd_imp_cart_vel_t.c_str() , "w" );
    
    ptr_file_impactveld = fopen ( savefile_impactveld.c_str() , "w" );
    
    ptr_file_alpha_fuzzy = fopen ( savefile_alpha_fuzzy.c_str() , "w" );
    
    ptr_file_dqcmd = fopen ( savefile_dqcmd.c_str() , "w" );
    
    ptr_file_qcmd = fopen ( savefile_qcmd.c_str() , "w" );
    
    ptr_file_qmsr = fopen ( savefile_qmsr.c_str() , "w" );
    
    ptr_file_wrenchmsr = fopen ( savefile_wrenchmsr.c_str() , "w" );
    
    ptr_file_wrenchEKF = fopen ( savefile_wrenchEKF.c_str() , "w" );
    
    ptr_file_torquesmsr = fopen ( savefile_torquesmsr.c_str() , "w" );
    
    ptr_file_torquesEKF = fopen ( savefile_torquesEKF.c_str() , "w" );
    
    ptr_file_impdamp_t = fopen ( savefile_impdamp_t.c_str() , "w" );
    
    ptr_file_impstiff_t = fopen ( savefile_impstiff_t.c_str() , "w" );
};

void SaveData_OSIFctrl::WriteData2File(std::array<double, 16> cart_pose, Eigen::VectorXd cmd_imp_cart_pose_t, Eigen::VectorXd cmd_cart_pose, Eigen::VectorXd cmd_imp_cart_vel_t, double impact_vel_d, double alpha_fuzzy, Eigen::VectorXd torquesmsr, Eigen::VectorXd torquesEKF, Eigen::VectorXd wrenchmsr, Eigen::VectorXd wrenchEKF, Eigen::VectorXd dqcmd, Eigen::VectorXd qcmd, Eigen::VectorXd qmsr, Eigen::VectorXd imp_damping_t, Eigen::VectorXd imp_stiffness_t, double time)
{ 
        
    fprintf ( ptr_file_cart_pose,"%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n", time, cart_pose[0], cart_pose[1], cart_pose[2], cart_pose[3], cart_pose[4], cart_pose[5], cart_pose[6], cart_pose[7], cart_pose[8], cart_pose[9], cart_pose[10], cart_pose[11], cart_pose[12], cart_pose[13], cart_pose[14], cart_pose[15] );  

    fprintf ( ptr_file_cmd_imp_cart_pose_t,"%f %f %f %f\n", time, cmd_imp_cart_pose_t(0), cmd_imp_cart_pose_t(1), cmd_imp_cart_pose_t(2));
    
    fprintf ( ptr_file_cmd_cart_pose,"%f %f %f %f %f %f %f\n", time, cmd_cart_pose(0), cmd_cart_pose(1), cmd_cart_pose(2), cmd_cart_pose(3), cmd_cart_pose(4), cmd_cart_pose(5));
    
    fprintf ( ptr_file_cmd_imp_cart_vel_t,"%f %f %f %f\n", time, cmd_imp_cart_vel_t(0), cmd_imp_cart_vel_t(1), cmd_imp_cart_vel_t(2));
    
    fprintf ( ptr_file_alpha_fuzzy,"%f %f\n", time, alpha_fuzzy);
    
    fprintf ( ptr_file_impactveld,"%f %f\n", time, impact_vel_d);
        
    fprintf ( ptr_file_dqcmd,"%f %f %f %f %f %f %f %f\n", time, dqcmd(0), dqcmd(1), dqcmd(2), dqcmd(3), dqcmd(4), dqcmd(5), dqcmd(6) );
    
    fprintf ( ptr_file_qcmd,"%f %f %f %f %f %f %f %f\n", time, qcmd(0), qcmd(1), qcmd(2), qcmd(3), qcmd(4), qcmd(5), qcmd(6) );
    
    fprintf ( ptr_file_qmsr,"%f %f %f %f %f %f %f %f\n", time, qmsr(0), qmsr(1), qmsr(2), qmsr(3), qmsr(4), qmsr(5), qmsr(6) );
    
    fprintf ( ptr_file_wrenchmsr,"%f %f %f %f %f %f %f\n", time, wrenchmsr(0), wrenchmsr(1), wrenchmsr(2), wrenchmsr(3), wrenchmsr(4), wrenchmsr(5) );
    
    fprintf ( ptr_file_wrenchEKF,"%f %f %f %f %f %f %f\n", time, wrenchEKF(0), wrenchEKF(1), wrenchEKF(2), wrenchEKF(3), wrenchEKF(4), wrenchEKF(5) );
    
    fprintf ( ptr_file_torquesmsr,"%f %f %f %f %f %f %f %f\n", time, torquesmsr(0), torquesmsr(1), torquesmsr(2), torquesmsr(3), torquesmsr(4), torquesmsr(5), torquesmsr(6) );
    
    fprintf ( ptr_file_torquesEKF,"%f %f %f %f %f %f %f %f\n", time, torquesEKF(0), torquesEKF(1), torquesEKF(2), torquesEKF(3), torquesEKF(4), torquesEKF(5), torquesEKF(6) );
    
    fprintf ( ptr_file_impdamp_t,"%f %f %f %f\n", time, imp_damping_t(0), imp_damping_t(1), imp_damping_t(2));
    
    fprintf ( ptr_file_impstiff_t,"%f %f %f %f\n", time, imp_stiffness_t(0), imp_stiffness_t(1), imp_stiffness_t(2));
        
    if ( ++elements_count > max_size_file && max_size_files_on==true )
    {
        fclose( ptr_file_cart_pose );
        
        fclose( ptr_file_cmd_imp_cart_pose_t );
        
        fclose( ptr_file_cmd_cart_pose );
        
        fclose( ptr_file_cmd_imp_cart_vel_t );
        
        fclose( ptr_file_impactveld );
        
        fclose( ptr_file_alpha_fuzzy );
        
        fclose( ptr_file_dqcmd );
        
        fclose( ptr_file_qcmd );
        
        fclose( ptr_file_qmsr );
        
        fclose( ptr_file_wrenchmsr );
        
        fclose( ptr_file_wrenchEKF );
        
        fclose( ptr_file_torquesmsr );
        
        fclose( ptr_file_torquesEKF );
        
        fclose( ptr_file_impdamp_t );
        
        fclose( ptr_file_impstiff_t );
        
        elements_count = 1;
    }

};

void SaveData_OSIFctrl::ClosingDataFiles(){

	fclose( ptr_file_cart_pose );
    
    fclose( ptr_file_cmd_imp_cart_pose_t );
    
    fclose( ptr_file_cmd_cart_pose );
    
    fclose( ptr_file_cmd_imp_cart_vel_t );
    
    fclose( ptr_file_impactveld );
    
    fclose( ptr_file_alpha_fuzzy );
    
    fclose( ptr_file_dqcmd );
    
    fclose( ptr_file_qcmd );
    
    fclose( ptr_file_qmsr );
        
    fclose( ptr_file_wrenchmsr );
    
    fclose( ptr_file_wrenchEKF );
    
    fclose( ptr_file_torquesmsr );
        
    fclose( ptr_file_torquesEKF );
    
    fclose( ptr_file_impdamp_t );
        
    fclose( ptr_file_impstiff_t );
}

}


