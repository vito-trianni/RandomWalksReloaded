#ifndef KILOBOT_RW_CONTROLLER_H
#define KILOBOT_RW_CONTROLLER_H

#include <argos3/core/control_interface/ci_controller.h>
#include <argos3/plugins/robots/generic/control_interface/ci_differential_steering_actuator.h>
#include <argos3/plugins/robots/generic/control_interface/ci_leds_actuator.h>
#include <argos3/plugins/robots/kilobot/control_interface/ci_kilobot_communication_actuator.h>
#include <argos3/plugins/robots/kilobot/control_interface/ci_kilobot_communication_sensor.h>

#include <argos3/core/utility/math/rng.h>
#include <argos3/core/utility/logging/argos_log.h>
#include <vector>

using namespace argos;

enum TStateNames {KILOBOT_STATE_STOP, KILOBOT_STATE_TURNING, KILOBOT_STATE_MOVING};

class CKilobotRWController : public CCI_Controller {

 public:

   CKilobotRWController();
   virtual ~CKilobotRWController() {}

   virtual void Init(TConfigurationNode& t_node);
   virtual void ControlStep();
   virtual void Reset();
   virtual void Destroy() {}

   //[DDS] Adding PDF from cdimidov Code]
   double wrapped_cauchy_ppf(const double c);
   double exponential_distribution( double lambda ); 
   double levy( const double c, const double alpha );

   // function to be used by the loop function to set the target
   // detection (a patch on the floor)
   void SetTargetDiscovered();
   void SetInformationReceived();

   inline const bool HasDiscoveredTarget() const {return m_bTargetDiscovered;}
   inline const bool HasReceivedInformation() const {return m_bInformationReceived;}
 private:

   /* Pointer to the LEDs actuator */
   CCI_LEDsActuator* m_pcLEDs;

   /* Pointer to the range and bearing actuator */
   CCI_KilobotCommunicationActuator*  m_pcRABA;

   /* Pointer to the range and bearing sensor */
   CCI_KilobotCommunicationSensor* m_pcRABS;

   /* Pointer to the differential steering actuator */
   CCI_DifferentialSteeringActuator* m_pcMotors;
   
   /* Parameters for random walk distributions */
   double m_fCRWExponent;
   double m_fLevyExponent;
   double m_fStdMotionSteps;

   /* Flag to check if the target was located/information was received */
   bool m_bTargetDiscovered;
   bool m_bInformationReceived;
       
   /* Store the time of individual discovery */
   UInt32 m_unTargetDiscovered;
   UInt32 m_unInformationReceived;

   /* Helper variable for kilo_ticks */
   UInt32 m_unKiloTicks;

   /* counters for random motion */
   UInt32 m_unTurningTicks;
   UInt32 m_unLastMotionTicks;
   UInt32 m_unStraightTicks;

   /* counter for maximum turning ticks, corresponding to 180 degree rotation */
   UInt8 m_unMaxTurningTicks;
    
   /* current/previous behavioural state (moving/turning) */
   TStateNames m_tCurrentState;
   TStateNames m_tPreviousState;
   
   /* actual motor speed */
   Real   m_fMotorL;
   Real   m_fMotorR;

   /* display color */
   CColor m_cCurrentColor;
   
   /* Message */
   message_t m_newMessage;
  
   /* random number generator */
   CRandom::CRNG*  m_pcRNG;
};

#endif
