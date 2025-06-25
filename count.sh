# for action in assemble unassemble clamp clampOnTable unclamp unclampOnTable screw unscrew; do
#     echo -n "$action: "
#     found=$(find /home/jyseo/hand_journal/data_MHAV/$action -type d -name RGB_undistorted 2>/dev/null)
#     if [ -n "$found" ]; then
#         du -ch $found 2>/dev/null | grep total$
#     else
#         echo "No matching RGB_undistorted folders"
#     fi
# done



# cd /home/jyseo/hand_journal/data_MHAV

# find . -type f -path "*/RGB_undistorted/*.jpg" | while read file; do
#     dir=$(dirname "$file")
#     ssh -p 22 jyseo@114.110.129.17 "mkdir -p /data/jyseo/data_MHAV/${dir#./}"
#     scp -P 22 "$file" jyseo@114.110.129.17:/data/jyseo/data_MHAV/${dir#./}/
# done

# cd /home/jyseo/hand_journal/data_MHAV

# find . -type f -path "*/RGB_undistorted/*.jpg" | while read file; do
#     dir=$(dirname "$file")
#     ssh -i ./AICA_088.pem -p 22 jyseo@114.110.129.17 "mkdir -p /data/jyseo/data_MHAV/${dir#./}"
#     rsync -avz -e "ssh -i ./AICA_088.pem -p 22" "$file" jyseo@114.110.129.17:/data/jyseo/data_MHAV/"$dir"/
# done


# cd /home/jyseo/hand_journal/data_MHAV

# find . -type f -path "**/RGB_undistorted_*.jpg" | while read file; do
#     [ -f "$file" ] || continue  # 실제 파일이 아닐 경우 skip

#     relative_dir=$(dirname "$file" | sed 's|^\./||')
#     target_dir="/data/jyseo/data_MHAV/$relative_dir"

#     ssh -i ./AICA_088.pem -p 22 jyseo@114.110.129.17 "mkdir -p \"$target_dir\""
#     rsync -avz -e "ssh -i ./AICA_088.pem -p 22" "$file" "jyseo@114.110.129.17:$target_dir/"
# done

# rsync -avz -e "ssh -i ./AICA_088.pem -p 22" \
#   --include='*/' \
#   --include='/home/jyseo/hand_journal/functional_hand_type' \
#   --exclude='*' \
#   ./ jyseo@114.110.129.17:/data/jyseo/functional_hand_type/


rsync -avz --exclude='*.pth' -e "ssh -i ./AICA_088.pem -p 22" \
  /home/jyseo/hand_journal/functional_hand_type/ \
  jyseo@114.110.129.17:/data/jyseo/functional_hand_type/