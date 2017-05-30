#include "kilobot_rw_lf.h"
#include "kilobot_rw_ctrl.h"

#include <argos3/core/simulator/simulator.h>
#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/plugins/robots/kilobot/simulator/kilobot_entity.h>
#include <argos3/plugins/robots/kilobot/simulator/kilobot_measures.h>

#include <set>
#include <iomanip>

const std::string CONFIGURATION_KILOBOT_RW_ARENA_RADIUS = "arena_radius";
const std::string CONFIGURATION_KILOBOT_RW_NUM_ROBOTS = "num_robots";
const std::string CONFIGURATION_KILOBOT_RW_TARGET_RADIUS = "target_radius";

/****************************************/
/****************************************/

CKilobotRWLoopFunctions::CKilobotRWLoopFunctions() :
   m_pcFloor(NULL),
   m_pcRNG(NULL),
   m_fArenaRadius(1),
   m_unNumRobots( 0 ),
   m_cTargetPosition(),
   m_fTargetRadius(0.1),
   m_tResults()
{
}

/****************************************/
/****************************************/

void CKilobotRWLoopFunctions::Init(TConfigurationNode& t_node) {
   /* Get a pointer to the floor entity */
   m_pcFloor = &GetSpace().GetFloorEntity();
   
   /* Create a new RNG */
   m_pcRNG = CRandom::CreateRNG("argos");

   /* Read parameters from configuration file */
   GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_ARENA_RADIUS, m_fArenaRadius, m_fArenaRadius );
   GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_NUM_ROBOTS, m_unNumRobots, m_unNumRobots );
   GetNodeAttributeOrDefault(t_node, CONFIGURATION_KILOBOT_RW_TARGET_RADIUS, m_fTargetRadius, m_fTargetRadius );

   ////////////////////////////////////////////////////////////////////////////////
   // CREATION AND POSITIONING OF THE ARENA WALLS (CIRCULAR ARENA)
   ////////////////////////////////////////////////////////////////////////////////
   Real wall_width = 0.01;
   Real wall_height = 0.05;

   std::ostringstream entity_id;
   Real m_unNumArenaWalls = 100;  // TODO: this can be mande a configurable parameter
   CRadians wall_angle = CRadians::TWO_PI/m_unNumArenaWalls;
   CVector3 wall_size(wall_width, 2.0*m_fArenaRadius*Tan(CRadians::PI/m_unNumArenaWalls), wall_height);
   for( UInt32 i = 0; i < m_unNumArenaWalls; i++ ) {
      entity_id.str("");
      entity_id << "wall_" << i;
      
      CRadians wall_rotation = wall_angle*i;
      CVector3 wall_position( m_fArenaRadius*Cos(wall_rotation), m_fArenaRadius*Sin(wall_rotation), 0 );
      CQuaternion wall_orientation;
      wall_orientation.FromEulerAngles( wall_rotation,  CRadians::ZERO, CRadians::ZERO );
      
      CBoxEntity* wall = new CBoxEntity( entity_id.str(), wall_position, wall_orientation, false, wall_size );
      AddEntity(*wall);
   }

   ////////////////////////////////////////////////////////////////////////////////
   // CREATION OF THE ROBOTS
   ////////////////////////////////////////////////////////////////////////////////
   std::ostringstream kilobot_id;
   CVector3 kilobot_position = CVector3(m_fArenaRadius,m_fArenaRadius,0); // init position must be out of the arena
   for( UInt32 i = 0; i < m_unNumRobots; i++ ) {
      kilobot_id.str("");
      kilobot_id << i;
      CKilobotEntity* kilobot = new CKilobotEntity( kilobot_id.str(), "rw", kilobot_position, CQuaternion() );
      AddEntity( *kilobot );
   }
   m_cKilobots = GetSpace().GetEntitiesByType("kilobot");

   // set the experiment
   SetExperiment();
}

/****************************************/
/****************************************/

void CKilobotRWLoopFunctions::Reset() {
   SetExperiment();
}

/****************************************/
/****************************************/

void CKilobotRWLoopFunctions::Destroy() {
}

/****************************************/
/****************************************/

CColor CKilobotRWLoopFunctions::GetFloorColor(const CVector2& c_position_on_plane) {
   if( (c_position_on_plane - m_cTargetPosition).Length() < m_fTargetRadius ) {
      return CColor::GRAY50;
   }
   return CColor::WHITE;
}

/****************************************/
/****************************************/

void CKilobotRWLoopFunctions::SetExperiment() {
   // initialise/reset the internal variables
   m_tResults.Reset();

   // initialise kilobots randomly
   for(CSpace::TMapPerType::iterator it = m_cKilobots.begin(); it != m_cKilobots.end(); ++it) {
      /* Get handle to kilobot entity and controller */
      CKilobotEntity& cKilobot = *any_cast<CKilobotEntity*>(it->second);      

      /* Get a random rotation within the circular arena */
      CQuaternion random_rotation;
      CRadians    random_rotation_angle(m_pcRNG->Uniform(CRange<Real>(-CRadians::PI.GetValue(),CRadians::PI.GetValue())));
      random_rotation.FromEulerAngles( random_rotation_angle,  CRadians::ZERO, CRadians::ZERO );

      /* Get a non-colliding random position within the circular arena */
      bool distant_enough = false;
      UInt32 m_unMaxInitTrials = 1000;
      UInt32 un_init_trials = 0;
      while( !distant_enough && (++un_init_trials < m_unMaxInitTrials) ) {
         Real rho = m_pcRNG->Uniform(CRange<Real>(0,m_fArenaRadius));
         Real theta = m_pcRNG->Uniform(CRange<Real>(-CRadians::PI.GetValue(),CRadians::PI.GetValue()));
         CVector3 random_position(rho*cos(theta), rho*sin(theta), 0);
         CVector3 kilobot_displacement( -KILOBOT_ECCENTRICITY, 0, 0);   
         distant_enough = MoveEntity( cKilobot.GetEmbodiedEntity(), random_position+kilobot_displacement.RotateZ(random_rotation_angle), random_rotation, false );
         if( un_init_trials > m_unMaxInitTrials ) {
            LOGERR << "Failed to move entity " << cKilobot.GetId() << " for  " <<  m_unMaxInitTrials << " trials" << std::endl;         
            LOGERR.Flush();
         }
      }
   }

   /* Get a  random position for the target */
   Real target_rho = m_pcRNG->Uniform(CRange<Real>(0,m_fArenaRadius-m_fTargetRadius));
   Real target_theta = m_pcRNG->Uniform(CRange<Real>(-CRadians::PI.GetValue(),CRadians::PI.GetValue()));
   m_cTargetPosition = CVector2(target_rho*cos(target_theta), target_rho*sin(target_theta));
}


/****************************************/
/****************************************/

void CKilobotRWLoopFunctions::PostStep() {
   UInt32 un_robot_index = 0;
   UInt32 num_robots_with_discovery = 0;
   UInt32 num_robots_with_info = 0;
   for(CSpace::TMapPerType::iterator it = m_cKilobots.begin(); it != m_cKilobots.end(); ++it, ++un_robot_index) {
      /* Get handle to kilobot entity and controller */
      CKilobotEntity& c_kilobot = *any_cast<CKilobotEntity*>(it->second);
      CVector2 c_kilobot_xy_position(c_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetX(),c_kilobot.GetEmbodiedEntity().GetOriginAnchor().Position.GetY());
      CKilobotRWController& c_controller = dynamic_cast<CKilobotRWController&>(c_kilobot.GetControllableEntity().GetController());

      if( c_controller.HasDiscoveredTarget() ) {
         num_robots_with_discovery += 1;
      }
      else if( (c_kilobot_xy_position - m_cTargetPosition).Length() < m_fTargetRadius) {
         c_controller.SetTargetDiscovered();
      }

      if( c_controller.HasReceivedInformation() ) {
         num_robots_with_info += 1;
      }
   }

   // Check results
   m_tResults.m_fFractionWithDiscovery = ((Real)num_robots_with_discovery)/((Real)m_unNumRobots);
   m_tResults.m_unFullInformationTime = ((Real)num_robots_with_info)/((Real)m_unNumRobots);
   
   if( (num_robots_with_discovery == m_unNumRobots) && (m_tResults.m_unFullDiscoveryTime == 0) )
      m_tResults.m_unFullDiscoveryTime = GetSpace().GetSimulationClock();
   
   if( (num_robots_with_info == m_unNumRobots) && (m_tResults.m_unFullInformationTime == 0) )
      m_tResults.m_unFullInformationTime = GetSpace().GetSimulationClock();
}



/****************************************/
/****************************************/

bool CKilobotRWLoopFunctions::IsExperimentFinished() {
   return (m_tResults.m_unFullDiscoveryTime != 0);
}

/****************************************/
/****************************************/


REGISTER_LOOP_FUNCTIONS(CKilobotRWLoopFunctions, "kilobot_rw_loop_functions")
