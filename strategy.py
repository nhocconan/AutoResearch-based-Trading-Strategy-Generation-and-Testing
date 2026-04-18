#!/usr/bin/env python3
"""
1d_Pivot_R1_S1_Breakout_VolumeTrend
1d strategy using daily Camarilla pivot levels (R1/S1) with volume confirmation and weekly trend filter.
- Long: Close breaks above daily R1 + volume > 1.5x weekly avg + weekly EMA34 > EMA89
- Short: Close breaks below daily S1 + volume > 1.5x weekly avg + weekly EMA34 < EMA89
- Exit: Opposite breakout or trend reversal
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
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
    
    # Get weekly data for trend filter and volume average
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA34 and EMA89 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Weekly volume average (10-period)
    vol_ma_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    rang = high_1d - low_1d
    r1 = close_1d + rang * 1.1 / 12
    s1 = close_1d - rang * 1.1 / 12
    
    # Align daily R1/S1 levels to 1d (no alignment needed as same timeframe)
    r1_aligned = r1  # Already on 1d timeframe
    s1_aligned = s1  # Already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # need enough for weekly EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_aligned[i] > ema_89_aligned[i]
        downtrend = ema_34_aligned[i] < ema_89_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakdown_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above daily R1
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below daily S1
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below daily S1
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above daily R1
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R1_S1_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0