cp ~/.bashrc ~/.bashrc_backup
cat ~/.bashrc_backup | sed -e '/acep_dbfilter/d' > ~/.bashrc
echo 'export PATH="$HOME/acep_dbfilter/dbfilter:$PATH"' >> ~/.bashrc
echo 'cd acep_dbfilter; git pull; cd ~' >> ~/.bashrc
