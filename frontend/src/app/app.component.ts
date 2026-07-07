import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { SegmentEvent, TranscriptionEvent, TranscriptionEventType } from './constants';
import { CommonModule } from '@angular/common';
import { animate, style, transition, trigger } from '@angular/animations';

@Component({
    selector: 'app-root',
    imports: [
        CommonModule,
        RouterOutlet
    ],
    templateUrl: './app.component.html',
    styleUrl: './app.component.scss',
    animations: [
        trigger('expandCollapse', [
            transition(':enter', [
            style({ height: '0px', opacity: 0 }),
            animate('300ms ease', style({ height: '*', opacity: 1 }))
            ]),
            transition(':leave', [
            style({ height: '*', opacity: 1 }),
            animate('300ms ease', style({ height: '0px', opacity: 0 }))
            ])
        ]),
    ]
})
export class AppComponent {
    title = 'SAR Radio Log';

    ws: WebSocket;

    segmentsShowing: SegmentEvent[] = [];

    isAudioIncoming: boolean = false;
    isConnectedToAudio: boolean = false;
    connectionText: string = "Connecting to audio...";

    constructor() {
        this.ws = new WebSocket(`ws://${location.hostname}:3011/transcription_stream`);
        this.ws.onmessage = (event: any) => {
            const transcriptionEvent: TranscriptionEvent = JSON.parse(event.data);
            console.log(transcriptionEvent);
            switch (transcriptionEvent.type) {
                case TranscriptionEventType.INITIALIZATION_EVENT:
                    // TODO anything to start transcription
                    this.connectionText = "Connected";
                    this.isConnectedToAudio = true;
                    break;
                case TranscriptionEventType.SPEECH_START_EVENT:
                    this.isAudioIncoming = true;
                    break;
                case TranscriptionEventType.SPEECH_STOP_EVENT:
                    this.isAudioIncoming = false;
                    break;
                case TranscriptionEventType.INFERENCE_START_EVENT:

                    break;
                case TranscriptionEventType.INFERENCE_STOP_EVENT:

                    break;
                case TranscriptionEventType.SEGMENT_EVENT:
                    this.segmentsShowing.unshift(transcriptionEvent as SegmentEvent);
                    break;
                default:
                    console.warn(`Unknown transcription event:`);
                    console.warn(transcriptionEvent);
                    break;
            }
        };
    }

    trackByMessageContent(index: number, item: any) {
        const entry: SegmentEvent = item as SegmentEvent;
        return `${entry.time}:${entry.text}`;
    }

}
