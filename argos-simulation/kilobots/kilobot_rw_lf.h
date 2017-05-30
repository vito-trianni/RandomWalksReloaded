#ifndef KILOBOT_RW_LF_H
#define KILOBOT_RW_LF_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/core/simulator/entity/floor_entity.h>
#include <argos3/core/utility/math/range.h>
#include <argos3/core/utility/math/rng.h>

#include <argos3/plugins/simulator/entities/box_entity.h>

using namespace argos;



////////////////////////////////////////////////////////////////
// Results struct
////////////////////////////////////////////////////////////////
struct TRWResults {
   UInt32 m_unFullDiscoveryTime;
   UInt32 m_unFullInformationTime;
   Real m_fFractionWithInformation;
   Real m_fFractionWithDiscovery;

   TRWResults() {
      m_unFullDiscoveryTime = 0;
      m_unFullInformationTime = 0;
      m_fFractionWithInformation = 0.0;
      m_fFractionWithDiscovery = 0.0;
   }

   void Reset() {
      m_unFullDiscoveryTime = 0;
      m_unFullInformationTime = 0;
      m_fFractionWithInformation = 0.0;
      m_fFractionWithDiscovery = 0.0;
   }

   friend std::ostream& operator<< ( std::ostream& os, const TRWResults& t_results ) {
      os << t_results.m_unFullDiscoveryTime << " " 
         << t_results.m_unFullInformationTime << " "
         << t_results.m_fFractionWithInformation << " "
         << t_results.m_fFractionWithDiscovery;
      return os;
   }
};

////////////////////////////////////////////////////////////////
// RW Loop Functions struct
////////////////////////////////////////////////////////////////

class CKilobotRWLoopFunctions : public CLoopFunctions {
 public:
   
   CKilobotRWLoopFunctions();
   virtual ~CKilobotRWLoopFunctions() {}

   virtual void Init(TConfigurationNode& t_tree);
   virtual void Reset();
   virtual void Destroy();
   virtual CColor GetFloorColor(const CVector2& c_position_on_plane);
   virtual void PostStep();

   virtual bool IsExperimentFinished();
   virtual void SetExperiment();

   const UInt32 GetNumRobots() const {return m_unNumRobots;};
   void SetNumRobots( const UInt32 un_num_robots ) {m_unNumRobots = un_num_robots;};

   inline const TRWResults& GetResults() const {return m_tResults;};

 private:
   CFloorEntity* m_pcFloor;
   CRandom::CRNG* m_pcRNG;

   Real m_fArenaRadius;
   UInt32 m_unNumRobots;
   CVector2 m_cTargetPosition;
   Real m_fTargetRadius;
   
   CSpace::TMapPerType m_cKilobots;
   TRWResults m_tResults;
};

#endif
