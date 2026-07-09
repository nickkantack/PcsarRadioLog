import sounddevice as sd

for i, dev in enumerate(sd.query_devices()):
    print(
        i,
        dev["name"],
        "inputs:", dev["max_input_channels"],
        "outputs:", dev["max_output_channels"],
        "default sr:", dev["default_samplerate"],
    )