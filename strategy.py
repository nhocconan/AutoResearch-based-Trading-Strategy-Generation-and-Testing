#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Donchian_Breakout_With_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 6h and weekly data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_6h) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot from previous week (H, L, C)
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 6h Donchian channel (20-period)
    def donchian_channels(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    upper_6h, lower_6h = donchian_channels(high, low, 20)
    
    # 6h volume filter: > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(upper_6h[i]) or 
            np.isnan(lower_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly R1 with volume confirmation and price above Donchian upper
            if (close[i] > r1_1w_aligned[i] and vol_filter[i] and 
                close[i] > upper_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S1 with volume confirmation and price below Donchian lower
            elif (close[i] < s1_1w_aligned[i] and vol_filter[i] and 
                  close[i] < lower_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below weekly pivot or Donchian lower
            if close[i] < pivot_1w_aligned[i] or close[i] < lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above weekly pivot or Donchian upper
            if close[i] > pivot_1w_aligned[i] or close[i] > upper_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot levels (R1/S1) act as strong support/resistance on 6s timeframe.
# Breakouts above R1 or below S1 with volume confirmation and Donchian breakout
# provide high-probability trend continuation trades.
# Weekly timeframe avoids noise, Donchian confirms breakout, volume filters false signals.
# Designed for 15-30 trades/year to minimize fee drag while capturing major moves.