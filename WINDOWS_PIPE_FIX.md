# Windows Pipe Fix Documentation

## Problem

When running `proctap --pid <PID> --stdout | proctap-whisper` on Windows, the following errors occurred:

```
[ERROR] Error writing to stdout: [Errno 22] Invalid argument
Exception ignored in: <_io.TextIOWrapper name='<stdout>' mode='w' encoding='cp932'>
OSError: [Errno 22] Invalid argument
```

Additionally, a Japanese error message appeared: "パイプを閉じています" (The pipe is being closed).

## Root Cause

On Windows, when binary data flows through a pipe (`proctap` sends binary audio data), the stdout handle is in a binary/pipe context. When `proctap-whisper` attempts to write **text** (transcriptions) to stdout in this context, Windows throws `[Errno 22] Invalid argument` because:

1. `proctap` pipes **binary audio data** through stdout
2. `proctap-whisper` reads this binary data successfully from stdin
3. But when `proctap-whisper` tries to write **text** to stdout, the text mode stdout can't handle being in a binary pipe context on Windows
4. This is a Windows-specific limitation with mixed binary/text pipe handling

## Solution

The fix automatically detects when stdout writing fails with an `OSError` and falls back to writing text output to **stderr** instead. This approach:

1. **Preserves binary pipe integrity** - Audio data continues to flow through stdout
2. **Transparently redirects text output** - User still sees transcriptions in terminal
3. **Maintains cross-platform compatibility** - No changes needed on Linux/macOS
4. **Follows Unix conventions** - Diagnostics and non-data output to stderr

## Changes Made

### 1. Modified `BasePipe.run_cli()` in `src/proctap_pipes/base.py`

Added try/except blocks around stdout writes for string/text results:

```python
if isinstance(result, str):
    try:
        output_stream.write(result)
        if not result.endswith("\n"):
            output_stream.write("\n")
    except OSError as e:
        # On Windows, writing text to stdout in a binary pipe context fails
        # Fall back to stderr for text output
        self.logger.debug(f"Failed to write to stdout ({e}), using stderr")
        sys.stderr.write(result)
        if not result.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.flush()
        continue
```

This pattern was applied to:
- String results in the main processing loop
- Generic object results (converted to string)
- Flush results (both string and generic objects)

### 2. Modified `main()` in `src/proctap_pipes/cli/whisper_cli.py`

Added try/except around the final flush print statement:

```python
result = pipe.flush()
if result:
    try:
        print(result)
    except OSError:
        # On Windows, writing to stdout in a binary pipe context fails
        # Write to stderr instead
        sys.stderr.write(result)
        if not result.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.flush()
```

### 3. Updated Documentation

- Added note in [README.md](README.md) explaining the behavior to users
- Included example of redirecting stderr to file: `proctap --pid 1234 --stdout | proctap-whisper 2> transcriptions.txt`
- Added to Design Principles: "Windows Compatibility: Automatic fallback to stderr for text output in binary pipe contexts"

## Testing

To test the fix:

```powershell
# This should now work without errors
proctap --pid <PID> --stdout | proctap-whisper

# Transcriptions will appear in terminal (via stderr)
# No more [ERROR] messages

# To save transcriptions to a file:
proctap --pid <PID> --stdout | proctap-whisper 2> transcriptions.txt
```

## Technical Details

### Why stderr?

1. **Unix Convention**: stderr is for diagnostics, logging, and non-data output
2. **Preserves stdout**: Binary audio data can continue flowing through stdout to other pipes
3. **Visible to user**: stderr output still appears in terminal by default
4. **Redirectable**: Users can redirect stderr separately if needed (`2>` operator)

### Cross-platform Behavior

- **Windows**: Text output automatically uses stderr when stdout fails
- **Linux/macOS**: Text output uses stdout normally (no stderr fallback triggered)
- **All platforms**: Binary data (audio passthrough) always uses stdout.buffer

### Error Handling Strategy

The fix uses a **graceful degradation** approach:
1. Try to write to stdout (normal behavior)
2. If OSError occurs, catch it and redirect to stderr
3. Log the issue at debug level for diagnostics
4. Continue processing without interruption

This ensures:
- No user-visible errors
- Transparent operation
- Maintains functionality across all platforms
- Easy to debug if needed (enable verbose logging)

## Future Considerations

Alternative approaches that were considered but not implemented:

1. **Force binary stdout**: Would break text-only pipes
2. **Separate output mode flag**: Adds complexity, not user-friendly
3. **Always use stderr**: Would break scripts expecting stdout
4. **Buffering workaround**: Doesn't solve the fundamental Windows limitation

The current solution (automatic fallback) provides the best balance of:
- Transparency to users
- Cross-platform compatibility
- Maintaining Unix philosophy
- Minimal code changes

## Related Issues

- Windows pipe handling: https://bugs.python.org/issue11395
- Text vs binary modes on Windows: https://docs.python.org/3/library/sys.html#sys.stdout
- Similar issues in other projects using mixed binary/text pipes on Windows
