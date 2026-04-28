#!/usr/bin/env python3
"""
12h_Pivot_Reversal_1dTrend_Volume
Hypothesis: Combines 1d trend filter with 12-hour pivot point reversal signals.
Uses volume confirmation to avoid false reversals. Designed for low trade frequency
(12-37 trades/year) to minimize fee burn while capturing mean-reversion moves
within the dominant daily trend. Works in both bull and bear markets by aligning
with higher timeframe trend while exploiting short-term exhaustion at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for pivot points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate classic pivot points from previous 12h bar
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    r2 = pivot + (high_12h - low_12h)
    s2 = pivot - (high_12h - low_12h)
    
    # Align to lower timeframe (12h) - values from previous 12h bar's close
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume confirmation: current volume > 1.5 * average volume (volume spike)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume spike
        vol_confirm = volume_spike[i]
        
        # Entry conditions: reversal at pivot levels with volume spike and trend alignment
        # Long: price drops to S1/S2 in uptrend with volume spike
        long_entry = ((close[i] <= s1_aligned[i]) or (close[i] <= s2_aligned[i])) and vol_confirm and uptrend
        # Short: price rises to R1/R2 in downtrend with volume spike
        short_entry = ((close[i] >= r1_aligned[i]) or (close[i] >= r2_aligned[i])) and vol_confirm and downtrend
        
        # Exit conditions: price returns to pivot level
        long_exit = close[i] >= pivot_aligned[i]
        short_exit = close[i] <= pivot_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Pivot_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0