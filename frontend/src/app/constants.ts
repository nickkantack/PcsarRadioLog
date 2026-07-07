
export interface TranscriptionEvent {

    type: string,
    time: string,
    sourceId: string

}

export class TranscriptionEventType {

    static readonly SEGMENT_EVENT: string = "SegmentEvent";
    static readonly INITIALIZATION_EVENT: string = "InitializationEvent";

}

export interface SegmentEvent extends TranscriptionEvent {

    text: string,
    isCutoff?: false,

}

export interface InitializationEvent extends TranscriptionEvent {

}