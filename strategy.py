#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with daily pivot point breakouts (S3/R3) filtered by weekly trend and volume.
# Uses pivot levels as dynamic support/resistance, weekly EMA for trend filter, and volume confirmation.
# Designed for fewer trades (target: 20-50/year) to avoid fee drag, works in both bull/bear markets via trend filter.
# Entry: Break of S3 (long) or R3 (short) with volume > 20-bar MA and price vs weekly EMA alignment.
# Exit: Opposite S1/R1 touch. Position size: 0.25 to limit drawdown.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (P, S1, S2, S3, R1, R2, R3)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    pivot_d = (high_d + low_d + close_d) / 3
    r1_d = 2 * pivot_d - low_d
    s1_d = 2 * pivot_d - high_d
    r2_d = pivot_d + (high_d - low_d)
    s2_d = pivot_d - (high_d - low_d)
    r3_d = high_d + 2 * (pivot_d - low_d)
    s3_d = low_d - 2 * (high_d - pivot_d)
    
    # Align to 12h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_1d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_1d, s3_d)
    r2_d_aligned = align_htf_to_ltf(prices, df_1d, r2_d)
    s2_d_aligned = align_htf_to_ltf(prices, df_1d, s2_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_d_aligned[i]) or np.isnan(s3_d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions: 
        # Long: break above daily S3 with upward trend and volume
        # Short: break below daily R3 with downward trend and volume
        long_breakout = close[i] > s3_d_aligned[i]
        short_breakout = close[i] < r3_d_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_up
        short_entry = short_breakout and vol_filter and trend_down
        
        # Exit conditions: opposite S1/R1 level touch
        long_exit = (close[i] < s1_d_aligned[i]) and position == 1
        short_exit = (close[i] > r1_d_aligned[i]) and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
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

name = "12h_DailyPivot_S3_R3_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0