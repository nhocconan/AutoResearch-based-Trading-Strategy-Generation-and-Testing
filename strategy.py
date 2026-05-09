#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above R1 with EMA50 uptrend and volume > 1.5x average
# Short when price breaks below S1 with EMA50 downtrend and volume > 1.5x average
# Exit when price retouches central pivot (PP) or reverses to opposite S3/R3
# Uses 12h timeframe to reduce trade frequency (target: 50-150 total trades over 4 years)
# Combines institutional support/resistance (Camarilla), trend (EMA), and conviction (volume)
# Designed for robustness in both bull and bear markets with controlled trade frequency

name = "12h_Camarilla_R1S1_Breakout_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    r3 = pp + (prev_high - prev_low) * 1.2500
    s1 = pp - (prev_high - prev_low) * 1.0833
    s3 = pp - (prev_high - prev_low) * 1.2500
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1, EMA50 uptrend, volume spike
            if (close[i] > r1_aligned[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, EMA50 downtrend, volume spike
            elif (close[i] < s1_aligned[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches central pivot or reverses to S3
            if (close[i] <= pp_aligned[i]) or (close[i] < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches central pivot or reverses to R3
            if (close[i] >= pp_aligned[i]) or (close[i] > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals