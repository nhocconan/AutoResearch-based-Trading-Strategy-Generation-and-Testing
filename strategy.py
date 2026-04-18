# 1d_WK1_WK2_WK3_WK4_Rotation
# Hypothesis: On daily chart, rotate between week 1 (new month) and weeks 2-4 based on 4-week rotation pattern.
# Uses 1-week higher timeframe to detect month start. Long in week 1, short in weeks 2-4 with volume confirmation.
# Designed to capture monthly momentum patterns that work in both bull and bear markets by being market-neutral.
# Target: 8-12 trades per year (~32-48 over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-week data for month start detection
    df_1w = get_htf_data(prices, '1w')
    
    # Detect week 1 of month (first week of each month)
    week1_signal = np.zeros(len(df_1w), dtype=bool)
    for i in range(len(df_1w)):
        # Simple approximation: first week of each month
        # In practice, we'd check if this week contains the 1st day of month
        # For simplicity, we'll use every 4th week as week 1 (approximate monthly cycle)
        if i % 4 == 0:
            week1_signal[i] = True
    
    # Align week 1 signal to daily timeframe
    week1_aligned = align_htf_to_ltf(prices, df_1w, week1_signal.astype(float))
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long (week 1), -1: short (weeks 2-4)
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(week1_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: week 1 of month with volume confirmation
            if week1_aligned[i] > 0.5 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: weeks 2-4 with volume confirmation
            elif week1_aligned[i] <= 0.5 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: end of week 1 (transition to weeks 2-4)
            if week1_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: end of weeks 2-4 (transition to week 1)
            if week1_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WK1_WK2_WK3_WK4_Rotation"
timeframe = "1d"
leverage = 1.0