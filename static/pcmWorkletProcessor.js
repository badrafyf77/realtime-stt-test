class PCMWorkletProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0];
    if (input) {
      const int16 = new Int16Array(input.length);
      for (let index = 0; index < input.length; index += 1) {
        let sample = input[index];
        sample = sample < -1 ? -1 : sample > 1 ? 1 : sample;
        int16[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PCMWorkletProcessor);
