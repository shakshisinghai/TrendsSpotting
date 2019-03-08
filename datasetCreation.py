
import numpy as np

import matplotlib.pyplot as plot
# Get x values of the sine wave

time        = np.arange(0, 50, 0.1);

# Amplitude of the sine wave is sine of a variable like time

a=np.random.randint(2,4,500)/10
amplitude   = 10+(a*time) + 10*np.sin(1.5*time)
# Plot a sine wave using time and amplitude obtained for the sine wave
plot.plot(time, amplitude)
# Give a title for the sine wave plot
print(int(amplitude[10]))
plot.title('Sine wave')

# Give x axis label for the sine wave plot
plot.xlabel('Time')
# Give y axis label for the sine wave 
plot.ylabel('Amplitude = sin(time)')

plot.grid(True, which='both')
plot.axhline(y=0, color='k')
plot.show()
# Display the sine wave

plot.show()
