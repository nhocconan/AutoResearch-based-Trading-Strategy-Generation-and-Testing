#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 12h/1d HTF - Camarilla pivot breakout with volume confirmation
    # Uses 12h Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) 
    # combined with 1d volume spike to catch institutional moves in both bull and bear markets
    # Target: 80-120 trades over 4 years (20-30/year) for optimal fee efficiency
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for HTF Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    r4_12h = close_12h + range_12h * 1.1 / 2
    r3_12h = close_12h + range_12h * 1.1 / 4
    s3_12h = close_12h - range_12h * 1.1 / 4
    s4_12h = close_12h - range_12h * 1.1 / 2
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or 
            np.isnan(s4_12h_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-day average (institutional participation)
        # Note: Using 1d volume for 6h bars - each 6h bar represents 1/4 of 1d volume
        volume_confirmed = volume_1d[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Breakout conditions: price breaks R4/S4 with volume
        breakout_up = close[i] > r4_12h_aligned[i]
        breakout_down = close[i] < s4_12h_aligned[i]
        
        # Mean reversion conditions: price touches R3/S3 with volume (fade extreme)
        touch_r3 = close[i] >= r3_12h_aligned[i]
        touch_s3 = close[i] <= s3_12h_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed
        enter_short = breakout_down and volume_confirmed
        exit_long_mean = position == 1 and touch_s3  # Exit long at S3 mean reversion
        exit_short_mean = position == -1 and touch_r3  # Exit short at R3 mean reversion
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and (exit_long_mean or close[i] < r3_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short_mean or close[i] > s3_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0