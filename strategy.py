#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot and R1/S1 from previous daily bar
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    R1_d = pivot_d + range_d
    S1_d = pivot_d - range_d
    
    # Align daily R1/S1 to 4h (wait for daily close)
    R1_d_aligned = align_htf_to_ltf(prices, df_1d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_1d, S1_d)
    pivot_d_aligned = align_htf_to_ltf(prices, df_1d, pivot_d)
    
    # Volume filter: current volume > 1.5 * 20-period average (~3.3 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 4h EMA50
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or
            np.isnan(pivot_d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_d_aligned[i]
        S1_val = S1_d_aligned[i]
        pivot_val = pivot_d_aligned[i]
        vol_filter = volume_filter[i]
        trend_up = close_val > ema_50[i]
        trend_down = close_val < ema_50[i]
        
        if position == 0:
            # Long: break above R1 with volume and uptrend
            if close_val > R1_val and vol_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and downtrend
            elif close_val < S1_val and vol_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot OR trend breaks
            if close_val < pivot_val or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot OR trend breaks
            if close_val > pivot_val or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals