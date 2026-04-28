#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Supertrend for trend filter (1w timeframe) with 1d Camarilla R3/S3 breakout and volume confirmation.
# Weekly Supertrend captures major trend regime (works in both bull/bear by filtering counter-trend noise).
# Breakout at 1d Camarilla R3/S3 levels (strong structure, fewer false breakouts than R1/S1).
# Volume spike (>2.0x 20-bar average) confirms breakout strength and reduces false signals.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h (within proven winning range).

name = "6h_Camarilla_R3S3_1wSupertrend_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend for trend filter (ATR=10, multiplier=3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2.0 + 3.0 * atr_10
    basic_lb = (high_1w + low_1w) / 2.0 - 3.0 * atr_10
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_1w)
    final_lb = np.zeros_like(close_1w)
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
        
        if supertrend[i-1] == final_ub[i-1]:
            if close_1w[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
                direction[i] = -1
        else:
            if close_1w[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
                direction[i] = 1
    
    # Align Supertrend direction to 6h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 1d Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d_prev) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels (R3/S3 provide good breakout structure with fewer false signals)
    R3 = pivot + range_1d * 1.1 / 4.0
    S3 = pivot - range_1d * 1.1 / 4.0
    
    # Align to 6h timeframe (use previous 1d bar's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume (stricter to reduce trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: Supertrend direction from 1w
        uptrend = supertrend_direction_aligned[i] == 1
        downtrend = supertrend_direction_aligned[i] == -1
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < S3_aligned[i] or not uptrend
        short_exit = close[i] > R3_aligned[i] or not downtrend
        
        # Handle entries and exits
        if long_breakout and uptrend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and downtrend and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals