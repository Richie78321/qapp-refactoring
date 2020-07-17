import numpy as np
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import copy

# CONSTANTS
_GHZ = 1.0
_MW_S1 = 'S1'  # disconnected for now
_MW_S2 = 'S2'  # channel 1, marker 1
_GREEN_AOM = 'Green'  # ch1, marker 2
_ADWIN_TRIG = 'Measure'  # ch2, marker 2
_WAVE = 'Wave'  # channel 1 and 2, analog I/Q data
_MARKER = 'Marker'
_CONN_DICT = {_MW_S1: None, _MW_S2: 1, _GREEN_AOM: 2, _ADWIN_TRIG: 4}
_PULSE_PARAMS = {
    'amplitude': 100, 'pulsewidth': 20, 'SB freq': 0.00, 'IQ scale factor': 1.0, 'phase': 0.0, 'skew phase': 0.0,
    'num pulses': 1
}
_DAC_UPPER = 1024.0  # DAC has only 1024 levels
_DAC_MID = 512
_IQTYPE = np.dtype('<f4')  # AWG520 stores analog values as 4 bytes in little-endian format
_MARKTYPE = np.dtype('<i1')  # AWG520 stores marker values as 1 byte


# TODO : Implement, rename, and place this method and other helper methods
def insert_multiple_pulses_into_event_dictionary(evt_dict, pulse, n=0):
    pass


class SequenceEvent(ABC):
    """Abstract base class for sequence events.

    :param event_type: The type of event.
    :param start: The start time of the event.
    :param stop: The stop time of the event.
    :param start_increment: The multiplier for incrementing the start time.
    :param stop_increment: The multiplier for incrementing the stop time.
    """

    def __init__(self, event_type, start, stop, start_increment=0, stop_increment=0):
        super().__init__()
        self.event_type = event_type
        self.start = start
        self.stop = stop
        self.start_increment = start_increment
        self.stop_increment = stop_increment

    def increment_time(self, dt=0):
        """Increments the start and stop times by dt.
        :param dt: The time increment.
        """

        self.start += dt * self.start_increment
        self.stop += dt * self.stop_increment

    @abstractmethod
    def generate_data(self):
        pass


class Channel:
    """Provides functionality for a sequence of :class:`sequence events <SequenceEvent>`.

    :param seq: A collection of :class:`sequence events <SequenceEvent>`.
    :param delay: Delay in the format [AOM delay, MW delay].
    :param pulse_params: A dictionary containing parameters for the pulse, containing: amplitude, pulseWidth, SB frequency, IQ scale factor, phase, skewPhase.
    :param connection_dict: A dictionary of the connections between AWG channels and switches/IQ modulators.
    :param timeres: The clock rate in ns.
    """

    def __init__(self, seq, delay=[0, 0], pulse_params=None, connection_dict=None, timeres=1):
        self.logger = logging.getLogger('seqlogger.seq_class')
        self.seq = seq
        self.timeres = timeres
        self.delay = delay

        # init the arrays
        self.wavedata = None
        self.c1markerdata = None
        self.c2markerdata = None

        # set the maximum length to be zero for now
        self.maxend = 0

        if pulse_params is None:
            self.pulse_params = self._PULSE_PARAMS
        else:
            self.pulse_params = pulse_params

        if connection_dict is None:
            self.connection_dict = self._CONN_DICT
        else:
            self.connection_dict = connection_dict

    def unpack_pulse_params(self, pulse_params):
        ssb_freq = float(pulse_params['SB freq']) * _GHZ  # SB freq is in units of GHZ
        iqscale = float(pulse_params['IQ scale factor'])
        phase = float(pulse_params['phase'])
        deviation = int(pulse_params['pulsewidth']) // self.timeres
        amp = int(pulse_params['amplitude'])  # needs to be a number between 0 and 100
        skew_phase = float(pulse_params['skew phase'])
        npulses = pulse_params['num pulses']
        return ssb_freq, iqscale, phase, deviation, amp, skew_phase, npulses

    def create_sequence(self, dt=0, custom_pulse_params = None):
        """Creates the data for the sequence.

        :param dt: Increment in time.
        :param custom_pulse_params: Use a custom set of pulse parameters instead of the default parameters.
        """

        # get the AOM delay
        aomdelay = int((self.delay[0] + self.timeres / 2) / self.timeres)  # proper way of rounding delay[0]/timeres
        self.logger.info("AOM delay is found to be %d", aomdelay)
        # get the MW delay
        mwdelay = int((self.delay[1] + self.timeres / 2) // self.timeres)
        self.logger.info("MW delay is found to be %d", mwdelay)

        # get all the pulse params
        ssb_freq, iqscale, phase, deviation, amp, skew_phase, npulses = self.unpack_pulse_params(self.pulse_params if custom_pulse_params is None else custom_pulse_params)

        # first increment the sequence by dt if needed
        for seq_event in self.seq:
            seq_event.increment_time(dt)

        # sort by start time
        self.seq.sort(key=lambda event: event.start)

        if any(seq_event.type == _WAVE for seq_event in self.seq):
            max_wave_duration = max((seq_event.end - seq_event.start) for seq_event in self.seq)
            # TODO : Work with lines 394-396

        # find max end time in sequence
        self.maxend = int(max(seq_event.end for seq_event in self.seq))

        # now we can init the arrays
        channel_data = np.zeros(shape=(2, 2, self.maxend), dtype=_MARKTYPE)
        wave_i = np.zeros(self.maxend, dtype=_IQTYPE)
        wave_q = np.zeros(self.maxend, dtype=_IQTYPE)
        for seq_event in self.seq:
            seq_data = seq_event.generate_data()

            # Do certain behaviors based on type of sequence event (there are currently two -- Waves and Markers)
            # In the case that this changes in the future, one might want to rethink the polymorphic structure instead
            # of extending a giant if-else block
            if seq_event.type == _WAVE:
                # Sequence event is wave
                i_data, q_data = seq_event.generate_iq(seq_data)
                wave_i[seq_event.start:seq_event.end] = i_data
                wave_q[seq_event.start:seq_event.end] = q_data
            elif seq_event.type == _MARKER:
                # Sequence event is a marker
                seq_event.generate_channel_data(channel_data)
            else:
                raise ValueError("Unsupported sequence event type: " + seq_event.type)

        # the marker data is simply the sum of the 2 markers since 1st bit represents m1 and 2nd bit represents m2
        # for each channel, and that's how we coded the Marker pulse class
        self.c1markerdata = channel_data[0][0] + channel_data[0][1]
        self.c2markerdata = channel_data[1][0] + channel_data[1][1]
        # the wavedata will store the data for the I and Q channels in a 2D array
        self.wavedata = np.array((wave_i, wave_q))

    def create_sequence_list(self, scan_param, start, steps, step_size):
        """Creates a list of sequences that scan over a defined pulse parameter.

        :param scan_param: The pulse parameter to scan over.
        :param start: The start of the parameter range.
        :param steps: The number of steps in the parameter range.
        :param step_size: The size of the steps in the parameter range.
        :return: Returns the list of sequences, each at a step in the parameter scan.
        """
        scan_list = np.arange(start=start, stop=start + (step_size * steps), step=step_size)

        # Create a list of sequences with variations depending on the scan parameter
        if scan_param == 'time':
            # Special case when scan_param is time
            sequence_list = [self.create_sequence(dt=x) for x in scan_list]
        else:
            # Ensure valid pulse param
            if scan_param not in self.pulse_params.keys():
                raise ValueError("Unknown pulse parameter used as scan parameter: " + scan_param)

            custom_pulse_params = self.pulse_params.deepcopy()
            sequence_list = []
            for x in scan_list:
                # TODO : Discuss possible special case with 'num pulses'
                custom_pulse_params[scan_param] = x
                sequence_list.append(self.create_sequence(custom_pulse_params=custom_pulse_params))

        return sequence_list


class WaveEvent(SequenceEvent, ABC):
    """The abstract base class for a pulse event. Specific pulses (like Gaussian) should implement this class.

    :param wave_type: The type of wave being implemented.
    :param start: The start time of the event.
    :param stop: The stop time of the event.
    :param num: The serial number of the pulse.
    :param width: How long the wave is going to be. It is an integer number representing samples.
    :param ssb_freq: The side band frequency, in order to get rid of the DC leakage from the mixer.
    :param iq_scale: The voltage scale for different channels (i.e. the for I and Q signals).
    :param phase: The phase difference between I and Q channels in degrees.
    :param skew_phase: Corrections to the phase in degrees.

    :type num: Integer
    :type ssb_freq: Floating point number
    :type iq_scale: Floating point number
    """

    def __init__(self, wave_type, start, stop, num, width, ssb_freq, iq_scale, phase, skew_phase):
        super().__init__(_WAVE, start, stop)
        self.vmax = 1.0  # The max voltage that AWG is using
        self.num = num
        self.wave_type = wave_type
        self.width = width
        self.ssb_freq = ssb_freq
        self.iq_scale = iq_scale
        self.phase = phase
        self.skew_phase = skew_phase

    def generate_iq(self, data):
        # This method is taking "envelope pulse data" and then adding the correction of IQ scale and phase to it. The
        # input is an array of floating point number. For example, if you are making a Gaussian pulse, this will be
        # an array with number given by exp(-((x-mu)/2*sigma)**2) It generates self.Q_data and self.I_data which will
        # be used to create waveform data in the .AWG file For all the pulse that needs I and Q correction,
        # the method needs to be called after you create the "raw pulse data"

        # Making I and Q correction
        temp_x = np.arange(self.width * 1.0)

        i_data = np.array(data * np.cos(2 * np.pi * (temp_x * self.ssb_freq + self.phase / 360.0)), dtype=_IQTYPE)
        q_data = np.array(data * np.sin(2 * np.pi * (temp_x * self.ssb_freq + self.phase / 360.0 + self.skew_phase / 360.0)) * self.iqscale, dtype=_IQTYPE)

        return i_data, q_data


class MarkerEvent(SequenceEvent, ABC):
    def __init__(self, marker_type, start, stop, num, width, marker_num, marker_on, marker_off):
        super().__init__(_MARKER, start, stop)
        self.marker_num = marker_num  # this number shows which marker we are using, 1 and 2 are for CH1 m1 and m2,
        # 3 and 4 are for CH2 m1 and m2, and so on
        self.marker_on = marker_on  # at which point you want to turn on the marker (can use this for marker delay)
        self.marker_off = marker_off  # at which point you want to turn off the marker

    @abstractmethod
    def generate_channel_data(self, data):
        pass

# TODO : Start defining the existing marker and wave classes

class Gaussian(WaveEvent):
    def __init__(self, num, width, ssb_freq, iq_scale, phase, deviation, amp, skew_phase=0):
        super().__init__(num, width, ssb_freq, iq_scale, phase, skew_phase)
        self.mean = self.width / 2.0  # The center of the Gaussian pulse
        self.deviation = deviation
        self.amp = amp * self.vmax/_DAC_UPPER # amp can be a value anywhere from 0 - 1000

    def generate_data(self):
        data = np.arange(self.width * 1.0, dtype=_IQTYPE)
        # making a Gaussian function
        data = np.float32(self.amp * np.exp(-((data - self.mean) ** 2) / (2 * self.deviation * self.deviation)))

        return data
