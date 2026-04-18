#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_with_Volume_and_Trend_Filter
Hypothesis: Weekly pivot point breakouts on daily chart with volume confirmation and 1w trend filter.
Uses weekly high/low from prior week as breakout levels. Volume spike confirms breakout strength.
Weekly EMA(34) filter ensures trading in direction of higher timeframe trend.
Designed for low trade frequency (7-25/year) to avoid fee drag while capturing significant moves.
Works in bull markets (breakouts above weekly high) and bear markets (breakdowns below weekly low).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points: use prior week's high and low as breakout levels
    # No calculation needed - direct use of weekly high/low
    weekly_high = high_1w  # Resistance level
    weekly_low = low_1w    # Support level
    
    # Align weekly levels to daily timeframe (wait for weekly bar to close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wh = weekly_high_aligned[i]
        wl = weekly_low_aligned[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above weekly high with volume spike and uptrend (price > weekly EMA)
            if price > wh and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume spike and downtrend (price < weekly EMA)
            elif price < wl and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below weekly low OR trend turns down
            if price < wl or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above weekly high OR trend turns up
            if price > wh or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_Breakout_with_Volume_and_Trend_Filter"
timeframe = "1d"
leverage = 1.0