#6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_HT
name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (PP, R1, R2, R3, S1, S2, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels
    r1 = pp + (prev_high - prev_low) * 1.0833
    r2 = pp + (prev_high - prev_low) * 1.1666
    r3 = pp + (prev_high - prev_low) * 1.2500
    s1 = pp - (prev_high - prev_low) * 1.0833
    s2 = pp - (prev_high - prev_low) * 1.1666
    s3 = pp - (prev_high - prev_low) * 1.2500
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, EMA34 uptrend, volume spike
            if (close[i] > r3_aligned[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, EMA34 downtrend, volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches central pivot or reverses to S1
            if (close[i] <= pp_aligned[i]) or (close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches central pivot or reverses to R1
            if (close[i] >= pp_aligned[i]) or (close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals