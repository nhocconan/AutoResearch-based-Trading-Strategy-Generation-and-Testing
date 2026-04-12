#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Trend_v1
Hypothesis: Uses 1-day Camarilla pivot levels with trend filter and volume confirmation.
Long when price breaks above H4 in uptrend; short when price breaks below L4 in downtrend.
Designed for low trade frequency by requiring breakout of key levels with trend alignment.
Works in bull via long breakouts, in bear via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (High + Low + Close) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    
    # Resistance levels: H1-H4
    h1 = close_1d + (range_ * 1.1 / 12)
    h2 = close_1d + (range_ * 1.1 / 6)
    h3 = close_1d + (range_ * 1.1 / 4)
    h4 = close_1d + (range_ * 1.1 / 2)
    
    # Support levels: L1-L4
    l1 = close_1d - (range_ * 1.1 / 12)
    l2 = close_1d - (range_ * 1.1 / 6)
    l3 = close_1d - (range_ * 1.1 / 4)
    l4 = close_1d - (range_ * 1.1 / 2)
    
    # Daily EMA50 for trend filter
    close_s = pd.Series(close_1d)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).values
    
    # Align to 12h
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below EMA50
        price_vs_ema = close[i] > ema50_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > h4_aligned[i] and price_vs_ema and vol_confirm
        breakout_short = close[i] < l4_aligned[i] and not price_vs_ema and vol_confirm
        
        # Exit when price returns to pivot or trend fails
        pivot_level = (high_1d + low_1d + close_1d) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(high_1d, pivot_level))
        exit_long = close[i] < pivot_aligned[i] or not price_vs_ema
        exit_short = close[i] > pivot_aligned[i] or price_vs_ema
        
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals