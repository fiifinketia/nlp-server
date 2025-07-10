#!/bin/bash

# Build script for NLP Server Singularity container
# Usage: ./build_singularity.sh [container_name]

set -e

# Default container name
CONTAINER_NAME=${1:-nlp-server.sif}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building NLP Server Singularity Container${NC}"
echo "=========================================="

# Check if Singularity is installed
if ! command -v singularity &> /dev/null; then
    echo -e "${RED}Error: Singularity is not installed${NC}"
    echo "Please install Singularity first:"
    echo "https://sylabs.io/guides/3.0/user-guide/installation.html"
    exit 1
fi

# Check if we're running as root (needed for some Singularity operations)
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root${NC}"
fi

# Build the container
echo -e "${GREEN}Building container: ${CONTAINER_NAME}${NC}"
echo "This may take a while..."

# Build with sudo if needed, otherwise without
if [ "$EUID" -eq 0 ]; then
    singularity build --fakeroot ${CONTAINER_NAME} nlp-server.def
else
    # Try without fakeroot first
    if ! singularity build ${CONTAINER_NAME} nlp-server.def 2>/dev/null; then
        echo -e "${YELLOW}Trying with fakeroot...${NC}"
        singularity build --fakeroot ${CONTAINER_NAME} nlp-server.def
    fi
fi

# Check if build was successful
if [ -f "${CONTAINER_NAME}" ]; then
    echo -e "${GREEN}✓ Container built successfully: ${CONTAINER_NAME}${NC}"
    
    # Show container info
    echo -e "\n${GREEN}Container Information:${NC}"
    singularity inspect ${CONTAINER_NAME}
    
    echo -e "\n${GREEN}Usage Examples:${NC}"
    echo "1. Run the server:"
    echo "   singularity run ${CONTAINER_NAME}"
    echo ""
    echo "2. Run with custom port:"
    echo "   singularity run ${CONTAINER_NAME} --port 8080"
    echo ""
    echo "3. Interactive shell:"
    echo "   singularity shell ${CONTAINER_NAME}"
    echo ""
    echo "4. Execute commands:"
    echo "   singularity exec ${CONTAINER_NAME} python3 test_server.py"
    echo ""
    echo "5. Mount custom models directory:"
    echo "   singularity run --bind /path/to/models:/app/models ${CONTAINER_NAME}"
    
else
    echo -e "${RED}✗ Container build failed${NC}"
    exit 1
fi 