# Remove roscpp loggers

client_file = '/usr/local/lib/python3.9/site-packages/rospy/client.py'

with open(client_file, 'r') as f:
  lines = f.readlines()
  f.seek(0)
  line_nums = list(range(67, 69)) + list(range(141, 151)) + list(range(160, 172))
  lines = [lines[i] for i in range(len(lines)) if i not in line_nums]

with open(client_file, 'w') as f:
  f.writelines(lines)
