#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1
Hypothesis: Weekly pivot levels (R1/S1) from the prior week act as strong support/resistance on the daily chart. 
Breakouts above R1 or below S1 with trend confirmation (price > weekly EMA34) capture sustained moves. 
Volume filter reduces false breakouts. Designed for ~15-25 trades/year on 1d timeframe to minimize fee drag.
Works in bull via trend-following breaks above R1 and in bear via breakdowns below S1.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and EMA
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to daily (already delayed by weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Weekly EMA34 for trend filter
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema34)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
            
        price = close[i]
        
        if position == 0:
            # Long: break above R1 with price above weekly EMA34 and volume confirmation
            if price > r1_aligned[i] and price > ema34_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with price below weekly EMA34 and volume confirmation
            elif price < s1_aligned[i] and price < ema34_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below pivot (support) or volume fails
            if price < pivot_aligned[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above pivot (resistance) or volume fails
            if price > pivot_aligned[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0