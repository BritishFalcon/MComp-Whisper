"""
This module is aimed at training a tokenizer for MIDI data. On research, I was able
to find a library, MidiTok, which provides a MIDI tokenizer structure. We will still
need to train it, but this makes life a lot easier.

Note we'll use "Tokenizer" rather than "Tokeniser" as the 'z' is most common in this space.
"""

# TODO: MidiTok
# https://github.com/Natooz/MidiTok

# REMI+ (MIDILike) is selected here to enable multiple instruments.
# TODO: Darin - perhaps have a look at the different tokenizers (see below) and see what you think
# https://miditok.readthedocs.io/en/latest/tokenizations.html
from miditok import MIDILike, TokenizerConfig
from pathlib import Path

# TODO: Darin - you may want to 'chime' in here, as I'm 'tone-deaf' to music data.
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, use_durations=True)
tokenizer = MIDILike(config)

midi_examples = [str(x) for x in Path("midi/").rglob("*.mid")]
tokenizer.train(vocab_size=30000, midi_files=midi_examples)
tokenizer.save(Path("tokenizer.json"))
