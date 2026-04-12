#!/usr/bin/env python3
"""
6h_12h_1d_Pivot_Reversal_v1
Hypothesis: On 6h timeframe, use 12h and 1d pivots to identify mean-reversion opportunities. 
Buy near 12h S1/S2 when 1d trend is up, sell near 12h R1/R2 when 1d trend is down.
Exit at 12h pivot point. Uses volume confirmation to avoid false signals.
Designed for low trade frequency (15-30/year) by requiring multiple confluence factors.
Works in bull/bear via 1d trend filter and mean-reversion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Pivot_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H PIVOT POINTS ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Standard pivot points (using previous 12h bar)
    pp_12h = (high_12h[:-1] + low_12h[:-1] + close_12h[:-1]) / 3
    r1_12h = 2 * pp_12h - low_12h[:-1]
    s1_12h = 2 * pp_12h - high_12h[:-1]
    r2_12h = pp_12h + (high_12h[:-1] - low_12h[:-1])
    s2_12h = pp_12h - (high_12h[:-1] - low_12h[:-1])
    
    # Prepend NaN for first bar (no previous data)
    pp_12h = np.concatenate([[np.nan], pp_12h])
    r1_12h = np.concatenate([[np.nan], r1_12h])
    s1_12h = np.concatenate([[np.nan], s1_12h])
    r2_12h = np.concatenate([[np.nan], r2_12h])
    s2_12h = np.concatenate([[np.nan], s2_12h])
    
    # === 1D TREND FILTER (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1d > ema_50
    trend_down = close_1d < ema_50
    
    # === VOLUME CONFIRMATION (20-period average) ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Align 12h data to 6h timeframe
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if indicators not available
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(r2_12h_aligned[i]) or 
            np.isnan(s2_12h_aligned[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Entry conditions: mean reversion at 12h S1/S2 with 1d trend
        long_setup = (close[i] <= s1_12h_aligned[i]) and trend_up_aligned[i] > 0.5 and vol_confirm
        short_setup = (close[i] >= r1_12h_aligned[i]) and trend_down_aligned[i] > 0.5 and vol_confirm
        
        # Exit conditions: return to 12h pivot point
        exit_long = close[i] >= pp_12h_aligned[i]
        exit_short = close[i] <= pp_12h_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals