import sys
import os
import subprocess

if __name__ == '__main__':
    n_runs = int(sys.argv[1])

    if len(sys.argv) > 2:
        prefix = sys.argv[2]
    else:
        prefix = 'adda'
    
    os.makedirs('./configs', exist_ok=True)
    os.makedirs('./routes', exist_ok=True)

    for i in range(1, n_runs + 1):
        with open(f'./{prefix}_{i}.log', 'r') as infile:
            lines = infile.readlines()
            config = lines[-4].split(' ')
            config[-1].strip('\n')
            config = '\n'.join(config)
            routes = lines[-2]

            with open(f'./configs/{prefix}_{i}.csv', 'a') as c_outfile:
                c_outfile.writelines(config)
            with open(f'./routes/{prefix}_{i}.csv', 'a') as r_outfile:
                r_outfile.write(routes.strip('\n'))
    
    subprocess.run(['tar', '-czvf', f'{prefix}.tar.gz', 'configs/', 'routes/'])
