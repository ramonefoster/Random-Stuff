import serial
import serial.tools.list_ports

import time

class COMSimulator():
    def __init__(self):        
        self._serial = None
        self._timeout = 2
        self._status = "       877 *000100000001100"
    
    def _ports(self):
        self.list = serial.tools.list_ports.comports()
        coms = []
        for element in self.list:
            coms.append(element.device)

        return(coms)

    def connect(self):
        if "COM2" in self._ports():
            self._serial = serial.Serial(
                port="COM2",
                baudrate=9600,                
                timeout=self._timeout ,
                write_timeout = 0.5
            )
            try:
                self._serial.close()
            except:
                print("Error closing")
            if not self._serial.is_open:
                try: 
                    self._serial.open()
                except Exception as e:
                    print(e)

    def disconnect(self):
        if self._serial.is_open:
            try:
                self._serial.close()
            except Exception as e:
                print(e)
    
    def run(self):
        while True:
            if self._serial.is_open:
                ack = self._serial.readline().decode()

                if "PROG STATUS" in ack:
                    self._serial.write(self._status.encode())
                
                if "PROG PARAR" in ack:
                    break
            
            time.sleep(.5)

        self.disconnect()

x = COMSimulator()
x.connect()
x.run()
