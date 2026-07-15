from collections import deque
import math
import numpy as np
from enum import Enum

mean_energy_buffer = deque(maxlen=50)

min_variance_ever_seen = None

class VadState(Enum):
    SpeechPresent = 1
    SpeechAbsent = 2


def get_vad_state(
    pcmf32,
    sample_rate,
    nugget_ms,
    is_speaking,
    verbose=False,
):
    pcmf32 = np.asarray(pcmf32, dtype=np.float32)

    nugget_samples = int(sample_rate * nugget_ms / 1000)
    num_whole_nuggets = len(pcmf32) // nugget_samples

    if num_whole_nuggets == 0:
        return VadState.SpeechAbsent

    nugget_energies = []

    for i in range(num_whole_nuggets):
        start = i * nugget_samples
        end = start + nugget_samples

        energy = np.sum(np.abs(pcmf32[start:end]))

        nugget_energies.append(energy)

    energies = np.array(nugget_energies)

    overall_mean_energy = np.mean(energies)

    mean_energy_buffer.append(overall_mean_energy)

    variance = np.sum(
        (energies - overall_mean_energy)**2
    )

    if verbose:
        print("var:", variance)

    """
    Latest VAD theory:
    Silence when connected to the radio produces a tight distribution; the bottom
    5% of energies for the whole VAD window are presumed to be mostly silence.
    Therefore, we'll keep a running buffer of mean energies for the whole VAD
    window and if the mean energy exceeds some margin above the bottom 5%, then
    there is speech.

    We choose the bottom 5% instead of the minimum to avoid a situation where a 
    freak incident causing very low mean energy and causes lots of blank audio
    until the buffer clears the freaky low energy. It also helps the threshold
    adapt a little more quickly if, for instance, the energy is drifting up.

    There is a risk that there truly won't be any silence for the duration of
    the buffer. This would cause dropped audio until enough blank audio rolls
    in to bring the bottom 5% to silence. Thus, don't make the buffer too small
    cause this would really stink.

    When the buffer is underfilled (shortly after startup), the logic below
    will essentially fallback to comparing mean energy against the smallest
    mean energy seen.
    """

    sorted_means = sorted(list(mean_energy_buffer))
    threshold = 2 * sorted_means[math.floor(0.05 * len(mean_energy_buffer))]

    # print(f"Comparing {overall_mean_energy} to {threshold}")

    return VadState.SpeechPresent if overall_mean_energy > threshold else VadState.SpeechAbsent
