#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1d HTF - Camarilla pivot breakout with volume confirmation
    # Uses 1d Camarilla levels (R3/S3 for fade, R4/S4 for breakout) with 6h volume spike
    # Designed to capture institutional breakouts in both bull and bear markets
    # Target: 50-150 trades over 4 years (12-37/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Classic Camarilla: pivot = (H+L+C)/3
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Calculate 6h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)  # Note: using 1d volume average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(r4_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Breakout conditions at R4/S4 (continuation)
        breakout_up = close[i] > r4_1d_aligned[i]
        breakout_down = close[i] < s4_1d_aligned[i]
        
        # Fade conditions at R3/S3 (mean reversion)
        fade_up = close[i] < r3_1d_aligned[i] and close[i] > s3_1d_aligned[i]  # Inside R3-S3
        fade_down = close[i] > s3_1d_aligned[i] and close[i] < r3_1d_aligned[i]  # Same as above
        
        # Entry logic: 
        # - Breakout continuation when volume confirms break of R4/S4
        # - Mean reversion fade when price reaches R3/S3 without volume confirmation
        enter_long = (breakout_up and volume_confirmed) or (close[i] <= s3_1d_aligned[i] and not volume_confirmed)
        enter_short = (breakout_down and volume_confirmed) or (close[i] >= r3_1d_aligned[i] and not volume_confirmed)
        
        # Exit conditions: 
        # - For breakout trades: exit at opposite Camarilla level (R3 for longs, S3 for shorts)
        # - For fade trades: exit at pivot
        exit_long = (position == 1 and 
                    ((breakout_up and volume_confirmed and close[i] <= r3_1d_aligned[i]) or
                     (not breakout_up and not volume_confirmed and close[i] >= pivot_1d_aligned[i])))
        exit_short = (position == -1 and 
                     ((breakout_down and volume_confirmed and close[i] >= s3_1d_aligned[i]) or
                      (not breakout_down and not volume_confirmed and close[i] <= pivot_1d_aligned[i])))
        
        # Need to align pivot as well
        pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1d_camarilla_breakout_fade_v1"
timeframe = "6h"
leverage = 1.0