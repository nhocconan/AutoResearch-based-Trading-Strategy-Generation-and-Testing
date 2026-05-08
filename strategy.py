#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1d trend filter (EMA34) and volume spike
# Long when price breaks above Camarilla R3 on 12h, 1d EMA34 rising, volume > 1.8x average
# Short when price breaks below Camarilla S3, 1d EMA34 falling, volume > 1.8x average
# Uses 12h for entry timing, 1d for trend filter to avoid whipsaws in choppy markets
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag and high win rate
# Camarilla pivot levels provide institutional-grade support/resistance
# EMA34 on 1d filters for primary trend direction, reducing false breakouts
# Volume spike confirms institutional participation in breakout

name = "12h_Camarilla_R3S3_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 12h OHLC
    # Using previous day's OHLC for current day's levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (P) = (H + L + C) / 3
    p = (high_12h + low_12h + close_12h) / 3.0
    
    # Calculate R3 and S3 levels
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    r3 = high_12h + 2 * (p - low_12h)
    s3 = low_12h - 2 * (high_12h - p)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Start from second bar to ensure we have previous 12h data for pivots
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Camarilla R3, 1d uptrend, volume spike
            if high_val > r3_val and ema34_1d_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S3, 1d downtrend, volume spike
            elif low_val < s3_val and ema34_1d_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or 1d trend down
            if low_val < s3_val or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or 1d trend up
            if high_val > r3_val or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals