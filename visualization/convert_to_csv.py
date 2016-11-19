#Get states and coordinates and generates a csv file

lats = []
lons = []
states = []
lat = 0.0
lon = 0.0

infile = 'all_states.json'

with open(infile, 'r') as f:
    for line in f:
        if ('coordinates' in line):
            lon = float(f.readline().strip().split(',')[0])
            lat = float(f.readline().strip().split(',')[0])
   
        #Pick only US states
        if ('state' in line):
            st = line.strip().split(':')[1]
            print(st)
            temp = f.readline()
            country = f.readline().split(',')[0]
            if ('country' in country):
                country = country.split(':')[1]
            else:
                country = ""
            if (len(st) < 40 and "United States of America" in country):
                states.append(st)
                lons.append(lon)
                lats.append(lat)

with open("geo.dat", "w") as f:
    f.write("state, lat, lon\n")
    tam = len(states)
    i = 0
    while (i < tam):
        line = states[i] + " " + str(lats[i]) + ", " + str(lons[i]) + "\n"
        f.write(line)
        i+=1

