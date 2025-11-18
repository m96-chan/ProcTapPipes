#!/bin/bash
# Test script to verify ProcTapPipes installation

echo "Testing ProcTapPipes installation..."
echo ""

# Test imports
echo "1. Testing Python imports..."
python3 -c "from proctap_pipes import BasePipe, WhisperPipe, LLMPipe, WebhookPipe; print('✓ All imports successful')" || exit 1

echo ""
echo "2. Testing CLI tools availability..."

# Check if CLI commands are registered
commands=("proctap-whisper" "proctap-llm" "proctap-webhook")

for cmd in "${commands[@]}"; do
    if command -v $cmd &> /dev/null; then
        echo "✓ $cmd is available"
    else
        echo "✗ $cmd not found (run 'pip install -e .' first)"
    fi
done

echo ""
echo "3. Testing help commands..."
proctap-whisper --help > /dev/null 2>&1 && echo "✓ proctap-whisper --help works" || echo "✗ proctap-whisper --help failed"
proctap-llm --help > /dev/null 2>&1 && echo "✓ proctap-llm --help works" || echo "✗ proctap-llm --help failed"
proctap-webhook --help > /dev/null 2>&1 && echo "✓ proctap-webhook --help works" || echo "✗ proctap-webhook --help failed"

echo ""
echo "Installation test complete!"
