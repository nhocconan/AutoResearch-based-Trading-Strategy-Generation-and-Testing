#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
# - Long when price breaks above R4 (Camarilla resistance level 4) from previous 1d AND volume > 1.5x 20-period average volume
# - Short when price breaks below S4 (Camarilla support level 4) from previous 1d AND volume > 1.5x 20-period average volume
# - Exit when price crosses back inside the Camarilla H-L range (between H3 and L3)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Camarilla pivots from 1d provide institutional support/resistance levels
# - Breakouts at R4/S4 indicate strong momentum with follow-through potential
# - Volume confirmation reduces false breakouts
# - Works in both bull and bear markets by capturing breakout moves in direction of trend

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
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations
    # Pivot point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1 = pp + (range_1d * 1.1 / 12)
    r2 = pp + (range_1d * 1.1 / 6)
    r3 = pp + (range_1d * 1.1 / 4)
    r4 = pp + (range_1d * 1.1 / 2)
    
    # Support levels
    s1 = pp - (range_1d * 1.1 / 12)
    s2 = pp - (range_1d * 1.1 / 6)
    s3 = pp - (range_1d * 1.1 / 4)
    s4 = pp - (range_1d * 1.1 / 2)
    
    # H3 and L3 for exit (price should stay between H3 and L3 when not breaking out)
    h3 = r3
    l3 = s3
    
    # Align HTF indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above R4 AND volume spike
            if (close[i] > r4_aligned[i] and volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below S4 AND volume spike
            elif (close[i] < s4_aligned[i] and volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside H3-L3 range
            exit_long = (position == 1 and close[i] < h3_aligned[i])
            exit_short = (position == -1 and close[i] > l3_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals