#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume confirmation and 1d EMA trend filter.
Long when price breaks above Camarilla R1 AND 4h volume > 1.5x 20-bar avg AND 1d EMA(50) rising.
Short when price breaks below Camarilla S1 AND 4h volume > 1.5x 20-bar avg AND 1d EMA(50) falling.
Exit when price touches Camarilla pivot point.
Uses 1h for execution, 4h for volume confirmation, 1d for EMA trend filter.
Designed to capture intraday breakouts with volume confirmation in trending markets.
Target: 15-30 trades/year per symbol (60-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_1d > np.roll(ema_50_1d, 1)
    ema_50_falling = ema_50_1d < np.roll(ema_50_1d, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: Pivot = (H+L+C)/3
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's data, so we shift by 1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    
    # First value will be invalid due to roll, set to nan
    close_1d_shift[0] = np.nan
    high_1d_shift[0] = np.nan
    low_1d_shift[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pivot = (high_1d_shift + low_1d_shift + close_1d_shift) / 3
    camarilla_r1 = close_1d_shift + (high_1d_shift - low_1d_shift) * 1.1 / 12
    camarilla_s1 = close_1d_shift - (high_1d_shift - low_1d_shift) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        # Need to get current 4h volume - we'll approximate using 1h volume scaled
        # Since we don't have direct 4h volume at 1h resolution, we use volume ratio
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        # Breakout conditions
        breakout_high = close[i] > camarilla_r1_aligned[i]
        breakout_low = close[i] < camarilla_s1_aligned[i]
        
        # Exit condition: touch pivot point
        touch_pivot = abs(close[i] - camarilla_pivot_aligned[i]) < 0.001 * close[i]  # within 0.1%
        
        if position == 0:
            # Long: break above R1 with volume confirmation and rising EMA
            if (breakout_high and volume_confirmed and ema_50_rising_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume confirmation and falling EMA
            elif (breakout_low and volume_confirmed and ema_50_falling_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: touch pivot point
            if touch_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: touch pivot point
            if touch_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Volume_1dEMA50_Trend"
timeframe = "1h"
leverage = 1.0