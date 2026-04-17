#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_1dEMA34_VolumeConfirm
Strategy: Camarilla pivot R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long: Price breaks above R1 + close > 1d EMA34 + volume > 1.5x 20-period average
Short: Price breaks below S1 + close < 1d EMA34 + volume > 1.5x 20-period average
Exit: Price returns to pivot point (PP) or volume drops below average
Position size: 0.25
Designed to capture breakouts with trend alignment and volume confirmation.
Timeframe: 4h
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day (OHLC)
    # Pivot points calculated from previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day: use first available values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate pivot point (PP)
    pp = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    range_prev = prev_high - prev_low
    r1 = pp + (range_prev * 1.1 / 12)
    s1 = pp - (range_prev * 1.1 / 12)
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1[i]) or 
            np.isnan(s1[i]) or np.isnan(pp[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Entry conditions
        if position == 0:
            # Long: Price breaks above R1 + trend up + volume
            if (close[i] > r1[i] and close[i] > ema_34_1d_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + trend down + volume
            elif (close[i] < s1[i] and close[i] < ema_34_1d_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to pivot point or volume drops
            if close[i] < pp[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to pivot point or volume drops
            if close[i] > pp[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0