from radar_params import RadarParams

if __name__ == '__main__':
    r = RadarParams(fc=77*1e12, S=30*1e6, Tc=40*1e-6,) 
    print(r.bandwidth)


