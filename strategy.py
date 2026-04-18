#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_VolumeTrend
4h strategy using daily Camarilla pivot R1/S1 breakouts with volume confirmation and 12h trend filter.
- Long: Close breaks above R1 + volume > 1.5x daily avg + 12h EMA34 > EMA89
- Short: Close breaks below S1 + volume > 1.5x daily avg + 12h EMA34 < EMA89
- Exit: Opposite breakout or trend reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
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
    
    # Get daily data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # Range = H - L
    range_1d = high_1d - low_1d
    # R1 = Pivot + (Range * 1.1 / 12)
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    # S1 = Pivot - (Range * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align daily R1/S1 to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    close_12h = df_12h['close'].values
    
    # 12h EMA34 and EMA89 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    ema_89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_89_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # need enough for EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(ema_89_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_12h_aligned[i] > ema_89_12h_aligned[i]
        downtrend = ema_34_12h_aligned[i] < ema_89_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_1d_aligned[i]
        breakdown_down = close[i] < s1_1d_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above R1
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below S1
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below S1
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above R1
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0