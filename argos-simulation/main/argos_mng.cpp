/**
 * @file <argos3/core/simulator/main.cpp>
 *
 * @author Carlo Pinciroli - <ilpincy@gmail.com>
 */

#include <argos3/core/simulator/simulator.h>
#include <argos3/core/utility/plugins/dynamic_loading.h>
#include <argos3/core/simulator/argos_command_line_arg_parser.h>
#include <kilobots/kilobot_rw_lf.h>
#include <iostream>

using namespace argos;

/**
 * @brief The standard main() function to run the ARGoS simulator.
 *
 * This main() function provides the basic logic to run the ARGoS
 * simulator: parses the command line, loads the experiment, runs the
 * simulation and disposes all the data.
 *
 * @param n_argc the number of command line arguments given at the shell.
 * @param ppch_argv the actual command line arguments.
 * @return 0 if everything OK; 1 in case of errors.
 */
int main(int n_argc, char** ppch_argv) {
   try {
      /* Create a new instance of the simulator */
      CSimulator& cSimulator = CSimulator::GetInstance();

      /* Configure the command line options */
      CARGoSCommandLineArgParser cACLAP;

      // argument -f - results filename
      std::string s_datafile = "results.dat";
      cACLAP.AddArgument<std::string>('f', "--file", "Specify the file to write results", s_datafile);
      
      // argument -r - number of runs to exectue
      UInt32 un_num_runs = 0;
      cACLAP.AddArgument<UInt32>('r', "--runs", "number of runs to be executed", un_num_runs);
      
      // argument -s - random seed for the experiments
      UInt32 un_seed = 0;
      cACLAP.AddArgument<UInt32>('s', "--seed", "random seed", un_seed);
      
      /* Parse command line */
      cACLAP.Parse(n_argc, ppch_argv);
      
      switch(cACLAP.GetAction()) {
      case CARGoSCommandLineArgParser::ACTION_RUN_EXPERIMENT: {
         CDynamicLoading::LoadAllLibraries();
         cSimulator.SetExperimentFileName(cACLAP.GetExperimentConfigFile());
         cSimulator.LoadExperiment();
         ////////////////////////////////////////////////////////////////////////////////
         // Start batch run
         ////////////////////////////////////////////////////////////////////////////////
         LOGERR << "WARNING: BATCH MODE - Use argos3 -c <config> for a full visualisation" << std::endl;
            
         std::ofstream of(s_datafile.c_str(),std::ios::out);
         CKilobotRWLoopFunctions* pcKilobotRWLF = &dynamic_cast<CKilobotRWLoopFunctions&>(cSimulator.GetLoopFunctions());
         
         // start evaluation
         for( UInt32 i = un_seed; i < un_seed+un_num_runs; ++i ) {
            // set the random seed in the simulator
            cSimulator.SetRandomSeed( i+1 );
            
            // resetting the experiment
            cSimulator.Reset();
            
            // run the simulation
            cSimulator.Execute();
            
            // retrieve the fitness values
            TRWResults results = pcKilobotRWLF->GetResults();
            of << results << std::endl;
         }
         of.close();
         break;
      }
      case CARGoSCommandLineArgParser::ACTION_SHOW_HELP:
         cACLAP.PrintUsage(LOG);
         break;
      case CARGoSCommandLineArgParser::ACTION_SHOW_VERSION:
         cACLAP.PrintVersion();
         break;
      case CARGoSCommandLineArgParser::ACTION_QUERY:
      case CARGoSCommandLineArgParser::ACTION_UNKNOWN:
         /* Should never get here */
         break;
      }
      /* Done, destroy stuff */
      cSimulator.Destroy();
   }
   catch(std::exception& ex) {
      /* A fatal error occurred: dispose of data, print error and exit */
      LOGERR << ex.what() << std::endl;
#ifdef ARGOS_THREADSAFE_LOG
      LOG.Flush();
      LOGERR.Flush();
#endif
      return 1;
   }
   /* Everything's ok, exit */
   return 0;
}
