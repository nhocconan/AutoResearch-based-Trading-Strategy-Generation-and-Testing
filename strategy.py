#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v1
# Hypothesis: Trade breakouts of weekly pivot levels (R1/S1, R2/S2) with 1d trend filter and volume confirmation.
# Weekly pivots act as strong support/resistance; breakouts often lead to sustained moves.
# 1d EMA50 filter ensures trading with higher timeframe trend.
# Volume confirmation reduces false breakouts.
# Works in bull/bear markets by capturing momentum after pivot level breaks.
# Target: 15-25 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot levels (from previous week)
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivots using previous week's OHLC (already complete in weekly data)
    # For weekly bar [i], we use OHLC from weekly bar [i-1] to avoid look-ahead
    # But since we get weekly data via get_htf_data, we need to shift the pivot calculation
    # Simpler: calculate pivots from current week's OHLC and use previous week's values
    # We'll calculate pivots for each weekly bar then shift by 1 to use previous week's
    
    # Extract weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot points for each weekly bar
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Shift to use previous week's pivots (avoid look-ahead)
    pivot = np.concatenate([[np.nan], pivot[:-1]])
    r1 = np.concatenate([[np.nan], r1[:-1]])
    s1 = np.concatenate([[np.nan], s1[:-1]])
    r2 = np.concatenate([[np.nan], r2[:-1]])
    s2 = np.concatenate([[np.nan], s2[:-1]])
    
    # Align to 6s timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily trend filter (1d EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or \
           np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(ema50_1d_6h[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_6h[i]
        daily_downtrend = close[i] < ema50_1d_6h[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: close below pivot or R1 (if we entered at R1/R2 breakout)
            if close[i] < pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above pivot or S1
            if close[i] > pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long breakout: price closes above R1 in uptrend
                if daily_uptrend and close[i] > r1_6h[i] and close[i-1] <= r1_6h[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below S1 in downtrend
                elif daily_downtrend and close[i] < s1_6h[i] and close[i-1] >= s1_6h[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals