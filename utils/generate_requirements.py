n = 12

config = 'train_environment.yml'
pip_config = 'requirements.txt'

with open(config, 'r') as f, open(pip_config, 'w') as out:
  lines = f.readlines()
  lines = lines[n:]
  lines = [line[4:] for line in lines]
  out.write('\n'.join(lines))

