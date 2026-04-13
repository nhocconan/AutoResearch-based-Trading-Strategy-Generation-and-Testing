#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot fade from 1d levels with volume confirmation
    # Fade at R3/S3 levels (mean reversion in range) with 1d volume spike confirmation
    # Breakout continuation at R4/S4 levels with volume confirmation
    # Uses 1d Camarilla levels for structure, 6h for execution
    # Discrete sizing 0.25 to limit fee churn
    # Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels use previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar will have NaN due to roll, that's handled by min_periods equivalent
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 4.0)
    s3 = pivot - (range_ * 1.1 / 4.0)
    r4 = pivot + (range_ * 1.1 / 2.0)
    s4 = pivot - (range_ * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d volume confirmation: volume > 1.3 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.3 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume_spike_aligned[i] > 0.5
        
        # Fade at R3/S3 (mean reversion in range)
        long_fade = (close[i] <= s3_aligned[i]) and vol_confirmed
        short_fade = (close[i] >= r3_aligned[i]) and vol_confirmed
        
        # Breakout continuation at R4/S4
        long_breakout = (close[i] > r4_aligned[i]) and vol_confirmed
        short_breakout = (close[i] < s4_aligned[i]) and vol_confirmed
        
        # Exit conditions: opposite Camarilla level or volume fade
        long_exit = (close[i] >= r3_aligned[i]) or (not vol_confirmed)
        short_exit = (close[i] <= s3_aligned[i]) or (not vol_confirmed)
        
        if (long_fade or long_breakout) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (short_fade or short_breakout) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_pivot_fade_v1"
timeframe = "6h"
leverage = 1.0