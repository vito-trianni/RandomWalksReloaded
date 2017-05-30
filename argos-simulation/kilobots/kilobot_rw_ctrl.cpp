#include "kilobot_rw_ctrl.h"

#include <argos3/core/utility/configuration/argos_configuration.h>
#include <argos3/core/utility/math/vector2.h>
#include <algorithm>

/****************************************/
/****************************************/

#define PIN_FORWARD 1.0f;
#define PIN_TURN    1.57f;
#define PIN_STOP    0.0f;

#define TARGET_INFO 250

//[DDS] Introducing same variables as cdimidov

CKilobotRWController::CKilobotRWController() :
   m_pcLEDs(NULL),
   m_pcRABA(NULL),
   m_pcRABS(NULL),
   m_pcMotors(NULL),
   m_fCRWExponent( 0.0 ),
   m_fLevyExponent(2.0),
   m_fStdMotionSteps(5*16),
   m_bTargetDiscovered(false),
   m_bInformationReceived(false),
   m_unTargetDiscovered( 0 ),
   m_unInformationReceived( 0 ),
   m_unKiloTicks( 0 ),
   m_unTurningTicks( 0 ),
   m_unLastMotionTicks( 0 ),
   m_unStraightTicks( 0 ),
   m_unMaxTurningTicks( 160 ),
   m_tCurrentState(KILOBOT_STATE_STOP),
   m_tPreviousState(KILOBOT_STATE_STOP),
   m_fMotorL(0.0f),
   m_fMotorR(0.0f),
   m_cCurrentColor(CColor::BLACK)
{
   m_pcRNG = CRandom::CreateRNG( "argos" );
}

/****************************************/
/****************************************/

void CKilobotRWController::Init(TConfigurationNode& t_node) {
   // Get sensor/actuator handles
   m_pcLEDs      = GetActuator<CCI_LEDsActuator>("leds");
   m_pcRABA      = GetActuator<CCI_KilobotCommunicationActuator>("kilobot_communication");
   m_pcRABS      = GetSensor  <CCI_KilobotCommunicationSensor>("kilobot_communication");
   m_pcMotors    = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");


   GetNodeAttributeOrDefault(t_node, "crw_exponent", m_fCRWExponent, m_fCRWExponent);
   GetNodeAttributeOrDefault(t_node, "levy_exponent", m_fLevyExponent, m_fLevyExponent);
   GetNodeAttributeOrDefault(t_node, "std_motion_steps", m_fStdMotionSteps, m_fStdMotionSteps);

   // reset/intialise the robot state
   Reset();
}

/****************************************/
/****************************************/

void CKilobotRWController::Reset() {    
   // reset/intialise the robot state
   m_bTargetDiscovered = false;
   m_bInformationReceived = false;
   m_unTargetDiscovered = 0;
   m_unInformationReceived = 0;
   
   m_unKiloTicks = 0;
   m_unTurningTicks = 0;
   m_unLastMotionTicks = 0;
   m_unStraightTicks = (UInt32) (fabs(levy(m_fStdMotionSteps,m_fLevyExponent)));
   m_unMaxTurningTicks = 160;   // TODO: this could be computed on the actual turning speed (PIN_TURN)

   m_tCurrentState = KILOBOT_STATE_MOVING;
   m_tPreviousState = KILOBOT_STATE_MOVING;
   m_fMotorL = m_fMotorR = PIN_FORWARD;
   
   m_cCurrentColor = CColor::WHITE;
   m_pcLEDs->SetAllColors(m_cCurrentColor);
}

/****************************************/
/****************************************/

double CKilobotRWController::wrapped_cauchy_ppf(const double c) {
   double val,theta,u,q;
   q = 0.5;
   u = m_pcRNG->Uniform(CRange<Real>(0.0,1.0));
   val = (1.0-c)/(1.0+c);
   theta = 2*atan(val*tan(M_PI*(u-q)));
   return theta;
}
  
/****************************************/
/****************************************/

double CKilobotRWController::exponential_distribution( double lambda ) {
   double u,x;
   u = m_pcRNG->Uniform(CRange<Real>(0.0,1.0));
   x = (-lambda)*log(1-u);
   return x;
}
   

/****************************************/
/****************************************/

double CKilobotRWController::levy( const double c, const double alpha ) {
   double u, v, t, s;
   
   u = M_PI * (m_pcRNG->Uniform(CRange<Real>(0.0,1.0)) - 0.5); /* uniform distribution */
   if( alpha == 1 ) {              /* cauchy case */
      t = tan( u );
      return c * t;
   }

   /* get a non-zero exponentially distributed value */
   do {
      v = exponential_distribution( 1.0 );
   }
   while (v == 0);

   if( alpha == 2 ) {            /* gaussian case */
      t = 2 * sin (u) * sqrt(v);
      return c * t;
   }

   /* general case */

   t = sin(alpha * u)/pow(cos(u), 1/alpha);
   s = pow(cos((1 - alpha)*u)/v, (1 - alpha)/alpha);
   return c * t * s;
}

/****************************************/
/****************************************/

void CKilobotRWController::SetTargetDiscovered() {
   if( !m_bTargetDiscovered ) {
      m_unTargetDiscovered = m_unKiloTicks;
      m_cCurrentColor = CColor::RED;
      m_pcLEDs->SetAllColors(m_cCurrentColor);
//       m_bTargetDiscovered = true; // If set here, LOG will not work as I want :D
   }
}


/****************************************/
/****************************************/

void CKilobotRWController::SetInformationReceived() {
   if( !m_bInformationReceived ) {
      m_unInformationReceived = m_unKiloTicks;
      if( !m_bTargetDiscovered ) {
         m_cCurrentColor = CColor::GREEN;
         m_pcLEDs->SetAllColors(m_cCurrentColor);
      }
      m_bInformationReceived = true;
   }
}

/****************************************/
/****************************************/

void CKilobotRWController::ControlStep() {
   ////////////////////////////////////////////////////////////////////////////////
   // compute the robot motion: according to PDF by cdimidov
   ////////////////////////////////////////////////////////////////////////////////
   m_unKiloTicks++;
   m_tPreviousState = m_tCurrentState;
   switch(m_tCurrentState) {
   case KILOBOT_STATE_TURNING:
      if( m_unKiloTicks > m_unLastMotionTicks + m_unTurningTicks ) {
         /* start moving forward */
         m_unLastMotionTicks = m_unKiloTicks;
         m_fMotorL = m_fMotorR = PIN_FORWARD;
         m_unStraightTicks = (UInt32)(fabs(levy(m_fStdMotionSteps,m_fLevyExponent)));
         m_tCurrentState = KILOBOT_STATE_MOVING;
      } 
      break;

   case KILOBOT_STATE_MOVING:
      if( m_unKiloTicks > m_unLastMotionTicks + m_unStraightTicks ) {
         /* perform a random turn */
         m_unLastMotionTicks = m_unKiloTicks;
	 UInt32 direction = m_pcRNG->Uniform(CRange<UInt32>(0,2));
         if( direction==0 ) {
            m_fMotorL = PIN_STOP;
            m_fMotorR = PIN_TURN;
         }
         else {
            m_fMotorL = PIN_TURN;
            m_fMotorR = PIN_STOP;
         }
         double angle = 0;
         if( m_fCRWExponent == 0 ) { // TODO: do we need this??
            angle = (m_pcRNG->Uniform(CRange<Real>(0.0,M_PI)));
         }     
	 else{
            angle = fabs(wrapped_cauchy_ppf(m_fCRWExponent));	 
         }
         m_unTurningTicks = (UInt32)((angle/M_PI) * m_unMaxTurningTicks ); 
	 m_tCurrentState = KILOBOT_STATE_TURNING;
         // printf("%u" "\n", straight_ticks);
      }        
      break;

   case KILOBOT_STATE_STOP:
   default:
      m_fMotorL = m_fMotorR = PIN_STOP;
      break;
   };

   m_pcMotors->SetLinearVelocity(m_fMotorL, m_fMotorR);
   
   
   ////////////////////////////////////////////////////////////////////////////////
   // Receive and transmit
   ////////////////////////////////////////////////////////////////////////////////
   if(m_unTargetDiscovered != 0 && !m_bTargetDiscovered){
      LOG << "Agent " << GetId() << ": target located at time " << m_unKiloTicks << std::endl;
      m_bTargetDiscovered = true;
   }

   // Check messages received from neighbours
   const CCI_KilobotCommunicationSensor::TPackets& tPackets = m_pcRABS->GetPackets();
   for( UInt32 i = 0; i < tPackets.size(); ++i ) {
      UInt8 flag = tPackets[i].Message->data[0];
      if( flag == TARGET_INFO && !m_bInformationReceived) {
         LOG << "Agent "<< GetId() << ": information received at time " << m_unKiloTicks<<std::endl;
	 SetInformationReceived();
      }
   }
   
   // Send messages to neighbours if target was located
   if( m_bTargetDiscovered || m_bInformationReceived ) {
      m_newMessage.type    = NORMAL;
      m_newMessage.data[0] = TARGET_INFO;
      // new_message.crc = message_crc(&new_message);
      
      m_pcRABA->SetMessage( &m_newMessage );
   }
}


/****************************************/
/****************************************/

REGISTER_CONTROLLER(CKilobotRWController, "kilobot_rw_controller")
