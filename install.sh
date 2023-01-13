# rm ~/dbfilter
cp ~/.bashrc ~/.bashrc_backup
cat ~/.bashrc_backup | sed -e '/dbfilter/d' > ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
cat ~/.bashrc
mv ~/.bashrc_backup ~/.bashrc
# cat .bashrc
