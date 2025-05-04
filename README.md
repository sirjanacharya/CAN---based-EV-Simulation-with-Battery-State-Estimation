In this project, a real-time SOC estimation framework is developed using MATLAB Simulink and CAN communication to emulate an electric vehicle environment. A standard drive cycle is used to control the motor, which draws power from the battery, and the resulting voltage, current, and timestamp data are transmitted via CAN bus to the Vehicle Control Unit (VCU). The CAN communication system is fully modeled using Simulink’s virtual CAN interface, incor
porating CAN Pack, CAN Transmit, CAN Receive, and CAN
Unpack blocks to replicate automotive-grade data exchange.
The message is then reformatted using a Byte Pack block
and transmitted to an external Python application over UDP.
At the VCU end, machine learning models are deployed to
estimate SOC from the received data. Several data-driven mod-
els, including LSTM, CNN-LSTM, and LSTM with attention
mechanism, are trained and evaluated on labeled EV battery
datasets. The comparative analysis reveals that the LSTM
with attention outperforms other models, achieving the highest
SOC estimation accuracy.
