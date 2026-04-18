# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_WeeklyBreakout_VolumeTrend
1d strategy using weekly support/resistance breakouts with volume confirmation and weekly trend filter.
- Long: Close breaks above weekly high + volume > 1.5x weekly avg + weekly EMA10 > EMA20
- Short: Close breaks below weekly low + volume > 1.5x weekly avg + weekly EMA10 < EMA20
- Exit: Opposite breakout or trend reversal
Designed for ~5-10 trades/year per symbol (20-40 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Support/Resistance levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly High and Low (resistance/support)
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Align weekly S/R levels to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Weekly EMA10 and EMA20 for trend filter
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Weekly volume average (10-period)
    vol_ma_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for EMA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_10_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_10_aligned[i] > ema_20_aligned[i]
        downtrend = ema_10_aligned[i] < ema_20_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > weekly_high_aligned[i]
        breakdown_down = close[i] < weekly_low_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above weekly high
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below weekly low
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below weekly low
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above weekly high
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBreakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0