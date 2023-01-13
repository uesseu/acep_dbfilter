cp ~/.bashrc ~/.bashrc_backup
cat ~/.bashrc_backup | sed -e '/dbfilter/d' > ~/.bashrc
echo 'export PATH="$HOME/acep_dbfilter/dbfilter:$PATH"' >> ~/.bashrc
