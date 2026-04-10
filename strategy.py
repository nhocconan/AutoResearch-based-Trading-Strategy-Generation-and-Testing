#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# - Long when price breaks above S3 (Camarilla support 3) AND volume > 1.8x 20-period average volume
# - Short when price breaks below R3 (Camarilla resistance 3) AND volume > 1.8x 20-period average volume
# - Exit when price crosses back inside the Camarilla H3-L3 range
# - Uses discrete position sizing 0.28 to balance return and fee drag
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Camarilla pivots from 1d provide institutional support/resistance levels
# - Volume confirmation reduces false breakouts
# - Works in both bull and bear markets by trading breakouts of key levels

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    rng = high_1d - low_1d
    
    # Camarilla levels
    # Resistance levels
    r4 = close_1d + rng * 1.5000
    r3 = close_1d + rng * 1.2500
    r2 = close_1d + rng * 1.1666
    r1 = close_1d + rng * 1.0833
    # Support levels
    s1 = close_1d - rng * 1.0833
    s2 = close_1d - rng * 1.1666
    s3 = close_1d - rng * 1.2500
    s4 = close_1d - rng * 1.5000
    
    # Align HTF indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3 = close_1d + rng * 1.0833  # H3 = R1
    l3 = close_1d - rng * 1.0833  # L3 = S1
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.28
            else:
                signals[i] = -0.28
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above S3 AND volume spike
            if (close[i] > s3_aligned[i] and volume_spike[i]):
                position = 1
                signals[i] = 0.28
            # Short conditions: price breaks below R3 AND volume spike
            elif (close[i] < r3_aligned[i] and volume_spike[i]):
                position = -1
                signals[i] = -0.28
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside the H3-L3 range
            exit_long = (position == 1 and close[i] < h3_aligned[i])
            exit_short = (position == -1 and close[i] > l3_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.28
                else:
                    signals[i] = -0.28
    
    return signals