
export interface TranscriptionEvent {

    event_type: string,
    time: string,
    sourceId: string

}

export class TranscriptionEventType {

    static readonly SEGMENT_EVENT: string = "SegmentEvent";
    static readonly INITIALIZATION_EVENT: string = "InitializationEvent";
    static readonly SPEECH_START_EVENT: string = "SpeechStartEvent";
    static readonly SPEECH_STOP_EVENT: string = "SpeechStopEvent";
    static readonly INFERENCE_START_EVENT: string = "InferenceStartEvent";
    static readonly INFERENCE_STOP_EVENT: string = "InferenceStopEvent";

}

export interface SegmentEvent extends TranscriptionEvent {

    text: string,
    audio: string,
    isCutoff?: false,

}

export interface InitializationEvent extends TranscriptionEvent {

}