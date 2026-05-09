#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above R3 with 12h EMA50 uptrend and volume > 2x average
# Short when price breaks below S3 with 12h EMA50 downtrend and volume > 2x average
# Exit when price retouches the central pivot (PP) or reverses to opposite S1/R1
# Uses tighter trend filter (12h EMA50) to reduce whipsaw and improve trend alignment
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
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
    
    # Calculate 12h Camarilla levels (PP, R1, R2, R3, S1, S2, S3)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla calculation
    prev_high = df_12h['high'].shift(1)
    prev_low = df_12h['low'].shift(1)
    prev_close = df_12h['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels
    r1 = pp + (prev_high - prev_low) * 1.0833
    r2 = pp + (prev_high - prev_low) * 1.1666
    r3 = pp + (prev_high - prev_low) * 1.2500
    s1 = pp - (prev_high - prev_low) * 1.0833
    s2 = pp - (prev_high - prev_low) * 1.1666
    s3 = pp - (prev_high - prev_low) * 1.2500
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp.values)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3.values)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1.values)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1.values)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3, 12h EMA50 uptrend, volume spike
            if (close[i] > r3_aligned[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 12h EMA50 downtrend, volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # EMA falling
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