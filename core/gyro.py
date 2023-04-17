import serial
import time
import numpy as np
from threading import Thread

class Gyroscope:
    class GyroscopeIniFailed(Exception):
        def __init__(self, msg):
            super().__init__(f"Gyroscope initialization failed! {msg}")
    
    def __init__(self, port_name):
        self.data = dict()
        try:
            self.serial_port = self.get_serial_port(port_name)
            # First update
            self._update_data()
            
        except Exception as e:
            raise self.GyroscopeIniFailed(str(e))
        
        self.data_thread = Thread(target=self.update_data)
        self.data_thread.daemon = True
        self.data_thread.start()

    def get_serial_port(self, name):
        ser = serial.Serial(
            port=name,
            baudrate=115200,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
            )
        assert(ser.isOpen())
        print(f"Port {ser.name} initialized!")
        return ser

    def update_data(self):
        while True:
            self._update_data()

    def _update_data(self):
        t0 = time.time()
        while True:
            # Ensure data is valid
            buf = self.serial_port.read(18)
            if buf[:2] != b'\xa5\xa5':
                if time.time() - t0 > 3:
                    raise Exception("Gyroscope get data timed out!")
                time.sleep(0.01)
                continue
            
            pitch = self.twosComplement_hex(buf[12] | buf[13] << 8) / 100
            yaw = self.twosComplement_hex(buf[10] | buf[11] << 8) / 100
            self.data.update({
                "X": self.twosComplement_hex(buf[2] | buf[3] << 8) / 100,
                "Y": self.twosComplement_hex(buf[4] | buf[5] << 8) / 100,
                "Z": self.twosComplement_hex(buf[6] | buf[7] << 8) / 100,

                "yaw_angle_velocity": self.twosComplement_hex(buf[8] | buf[9] << 8) / 100,
                "yaw_angle": yaw,
                "pitch_angle": pitch,
                "towards_direction": self.direction_unit_vector(pitch, yaw),
                "roll_angle": self.twosComplement_hex(buf[14] | buf[15] << 8) / 100,
                "checksum": (buf[16] | buf[17]) << 8
            })
            break

    @staticmethod
    def twosComplement_hex(hexval):
        bits = 16
        val = hexval
        if val & (1 << (bits-1)):
            val -= 1 << bits
        return val

    @staticmethod
    def direction_unit_vector(angle_yz, angle_zx):
        
        # Invert yaw angle, left hand coordinate system
        angle_zx = -angle_zx
        
        # Convert angles from degrees to radians
        angle_yz_rad = np.deg2rad(angle_yz)
        angle_zx_rad = np.deg2rad(angle_zx)

        # Calculate the components of the direction unit vector
        x = -np.sin(angle_zx_rad)  # Invert the x component for left-hand coordinate system
        y = np.sin(angle_yz_rad)
        z = np.sqrt(1 - x**2 - y**2)

        # Create the unit vector
        unit_vector = np.array([x, y, z])

        return unit_vector
