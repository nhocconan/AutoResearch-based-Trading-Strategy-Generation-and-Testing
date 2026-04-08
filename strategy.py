#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_volume_v1
# Hypothesis: On 6h timeframe, use weekly pivot points for directional bias and breakout confirmation with volume.
# Long when price breaks above weekly R3 with volume > 1.5x average and weekly trend up.
# Short when price breaks below weekly S3 with volume > 1.5x average and weekly trend down.
# Exit when price reverses back to weekly pivot or opposite signal triggers.
# Uses institutional pivot levels as support/resistance with volume confirmation to avoid false breakouts.
# Target: 15-25 trades/year to minimize fee decay while capturing institutional flow.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (HTF)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot = (high_weekly + low_weekly + close_weekly) / 3
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Weekly trend: price above/below weekly pivot
    weekly_uptrend = close > pivot_aligned
    weekly_downtrend = close < pivot_aligned
    
    # Volume confirmation: 24-period average on 6h (4 days)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(pivot_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to weekly pivot or opposite signal
            if close[i] <= pivot_aligned[i] or \
               (close[i] < s3_aligned[i] and volume[i] > 1.5 * avg_volume[i] and weekly_downtrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly pivot or opposite signal
            if close[i] >= pivot_aligned[i] or \
               (close[i] > r3_aligned[i] and volume[i] > 1.5 * avg_volume[i] and weekly_uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above weekly R3 with volume and weekly uptrend
            if close[i] > r3_aligned[i] and volume_ok and weekly_uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly S3 with volume and weekly downtrend
            elif close[i] < s3_aligned[i] and volume_ok and weekly_downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals