#!/bin/bash
echo "Quick Commands for ROCm Jupyter:"
echo "================================="
echo ""
echo "1. Start:          docker compose up -d"
echo "2. Stop:           docker compose down"
echo "3. Restart:        docker compose restart"
echo "4. Rebuild:        docker compose up --build -d"
echo "5. Logs:           docker compose logs -f"
echo "6. Shell:          docker exec -it rocm-jupyter bash"
echo "7. Check GPU:      docker exec rocm-jupyter rocm-smi"
echo "8. Backup:         tar -czf backup_$(date +%Y%m%d).tar.gz workspace/"
echo "9. Clean Docker:   docker system prune -a"
echo 'A. Dockser stats:  docker stats rocm-jupyter --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}{{.NetIO}}\t{{.BlockIO}}"'
echo 'B. Docker rmi   :  docker rmi rocm-pytorch-jupyter:latest 2>/dev/null || true' 
echo ""
echo "Access Jupyter:    http://$(hostname -I | awk '{print $1}'):8888"
echo "Token:             ********"
