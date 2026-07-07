import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { SegmentEvent, TranscriptionEvent, TranscriptionEventType } from './constants';
import { CommonModule } from '@angular/common';

@Component({
    selector: 'app-root',
    imports: [
        CommonModule,
        RouterOutlet
    ],
    templateUrl: './app.component.html',
    styleUrl: './app.component.scss'
})
export class AppComponent {
    title = 'SAR Radio Log';

    ws: WebSocket;

    segmentsShowing: SegmentEvent[] = [];

    constructor() {
        this.ws = new WebSocket(`ws://${location.hostname}:3011/transcription_stream`);
        this.ws.onmessage = (event: any) => {
            const transcriptionEvent: TranscriptionEvent = JSON.parse(event.data);
            console.log(transcriptionEvent);
            switch (transcriptionEvent.type) {
                case TranscriptionEventType.INITIALIZATION_EVENT:
                    // TODO anything to start transcription
                    break;
                case TranscriptionEventType.SEGMENT_EVENT:
                    this.segmentsShowing.push(transcriptionEvent as SegmentEvent);
                    break;
                default:
                    console.warn(`Unknown transcription event:`);
                    console.warn(transcriptionEvent);
                    break;
            }
        };
    }


}
