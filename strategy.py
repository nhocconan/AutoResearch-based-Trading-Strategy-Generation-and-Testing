#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts above R1 or below S1 with volume confirmation
capture momentum moves in both bull and bear markets, while fade at R2/S2 provides mean reversion in ranging conditions.
Targets 12-37 trades/year with controlled turnover.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, R2=P+(H-L), S2=P-(H-L)"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pivot, r1, s1, r2, s2 = calculate_pivot_points(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 week for no look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 or volume drops significantly
            if close[i] < s1_aligned[i] or volume[i] < (vol_ma[i] * 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 or volume drops significantly
            if close[i] > r1_aligned[i] or volume[i] < (vol_ma[i] * 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout: price breaks above R1 with volume spike
            if close[i] > r1_aligned[i] and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S1 with volume spike
            elif close[i] < s1_aligned[i] and vol_spike[i]:
                position = -1
                signals[i] = -0.25
            # Long mean reversion: price touches S2 and starts bouncing with volume
            elif close[i] <= s2_aligned[i] * 1.005 and close[i] > s2_aligned[i] and vol_spike[i]:
                # Only take if we're not in strong downtrend (price above 20-period MA)
                ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values[i]
                if not np.isnan(ma_20) and close[i] > ma_20:
                    position = 1
                    signals[i] = 0.25
            # Short mean reversion: price touches R2 and starts reversing with volume
            elif close[i] >= r2_aligned[i] * 0.995 and close[i] < r2_aligned[i] and vol_spike[i]:
                # Only take if we're not in strong uptrend (price below 20-period MA)
                ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values[i]
                if not np.isnan(ma_20) and close[i] < ma_20:
                    position = -1
                    signals[i] = -0.25
    
    return signals