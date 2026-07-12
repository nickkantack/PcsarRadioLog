import numpy as np
from enum import Enum


class VadState(Enum):
    SpeechPresent = 1
    SpeechAbsent = 2


def get_vad_state(
    pcmf32,
    sample_rate,
    nugget_ms,
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

        if verbose:
            print(energy)

    energies = np.array(nugget_energies)

    overall_mean_energy = np.mean(energies)

    variance = np.sum(
        (energies - overall_mean_energy) ** 2
    )

    if verbose:
        print("energies:", energies)
        print("var:", variance)

    return (
        VadState.SpeechPresent
        if variance > 90
        else VadState.SpeechAbsent
    )