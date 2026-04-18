#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Tight
Hypothesis: Trade breakouts above/below daily Camarilla R1/S1 levels in direction of weekly EMA(34) trend, confirmed by volume >2x 24-period average. Uses stricter volume filter (2x) and only allows entries on new highs/lows (breakout confirmation) to reduce trades. Position size 0.25 targeting ~20 trades/year to minimize fee drag. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1d calculations (previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's OHLC (completed day)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Daily Camarilla levels (based on previous day)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        range_ = prev_high_1d[i] - prev_low_1d[i]
        R1[i] = prev_close_1d[i] + range_ * 1.1 / 12
        S1[i] = prev_close_1d[i] - range_ * 1.1 / 12
    
    # 1w EMA trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align daily Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Align weekly EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2x 24-period average (stricter)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Breakout confirmation: price must make new 4-period high/low
    high_4 = np.full_like(high, np.nan)
    low_4 = np.full_like(low, np.nan)
    
    for i in range(4, len(high)):
        high_4[i] = np.max(high[i-4:i])
        low_4[i] = np.min(low[i-4:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period, 4)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(high_4[i]) or np.isnan(low_4[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation (stricter: 2x average)
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Breakout confirmation: new 4-period high/low
        breakout_high = high[i] > high_4[i]
        breakout_low = low[i] < low_4[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, breakout confirmation, and above weekly EMA
            if breakout_high and close[i] > R1_aligned[i] and vol_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, breakout confirmation, and below weekly EMA
            elif breakout_low and close[i] < S1_aligned[i] and vol_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reverse signal) or below weekly EMA
            if close[i] < S1_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (reverse signal) or above weekly EMA
            if close[i] > R1_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Tight"
timeframe = "12h"
leverage = 1.0