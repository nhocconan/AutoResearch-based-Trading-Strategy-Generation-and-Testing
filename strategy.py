#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 1d EMA50 trend filter.
Long when price breaks above R1 AND 4h volume > 1.5x 20-bar avg AND 1d EMA(50) rising.
Short when price breaks below S1 AND 4h volume > 1.5x 20-bar avg AND 1d EMA(50) falling.
Exit when price touches Camarilla midpoint (M).
Uses 4h for execution and volume, 1d for EMA trend filter.
Designed to capture strong intraday trends with volume confirmation across bull and bear markets.
Target: 20-50 trades/year per symbol.
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
    
    # Get 4h data for Camarilla pivot calculation and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Camarilla pivot levels (using previous day's OHLC)
    # We need to align 1d OHLC to 4h bars
    # For each 4h bar, use previous completed 1d bar's OHLC
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # M = (high + low + close)/3
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * range_1d / 12
    camarilla_s1 = close_1d - 1.1 * range_1d / 12
    camarilla_m = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_m_aligned = align_htf_to_ltf(prices, df_1d, camarilla_m)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_m_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_r1 = close[i] > camarilla_r1_aligned[i]
        breakdown_s1 = close[i] < camarilla_s1_aligned[i]
        
        # Exit condition: touch midpoint (M)
        touch_mid = abs(close[i] - camarilla_m_aligned[i]) < 0.001 * close[i]  # within 0.1%
        
        if position == 0:
            # Long: break above R1 with volume confirmation and rising EMA
            if (breakout_r1 and volume_confirmed and ema_50_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and falling EMA
            elif (breakdown_s1 and volume_confirmed and ema_50_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint (M)
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint (M)
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0