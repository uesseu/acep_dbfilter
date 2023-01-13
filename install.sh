cp ~/.bashrc ~/.bashrc_backup
cat ~/.bashrc_backup | sed -e '/acep_dbfilter/d' > ~/.bashrc
echo 'export PATH="$HOME/acep_dbfilter/dbfilter:$PATH"' >> ~/.bashrc
echo 'git pull acep_dbfilter' >> ~/.bashrc
