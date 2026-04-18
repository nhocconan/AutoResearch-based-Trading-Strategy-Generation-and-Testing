#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1
Hypothesis: Weekly pivot R1/S1 levels act as strong support/resistance. Breakouts above R1 or below S1 with trend confirmation (price > weekly EMA34) capture sustained moves. Weekly timeframe reduces noise, daily provides timely entries. Designed for ~15 trades/year to minimize fee drag.
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
    
    # Weekly data for pivot points and EMA
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly levels to daily
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_1w_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R1 with trend filter (price > EMA34) and volume
            if price > r1 and price > ema34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with trend filter (price < EMA34) and volume
            elif price < s1 and price < ema34 and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price falls below pivot or trend fails
            if price < pivot_aligned[i] or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price rises above pivot or trend fails
            if price > pivot_aligned[i] or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0