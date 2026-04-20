#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Trend
Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX filter for trend strength.
Goes long on breakout above upper band with volume > 1.5x average and 1d ADX > 25.
Goes short on breakdown below lower band with volume > 1.5x average and 1d ADX > 25.
Uses 12h for signal generation, 1d for trend filter to reduce whipsaw.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "12h_Donchian20_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Plus Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilder_smoothing(tr, period)
        plus_di = 100 * wilder_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilder_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian(20) on 12h
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    upper_band, lower_band = calculate_donchian(high, low, 20)
    
    # Calculate volume average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band AND volume > 1.5x average AND 1d ADX > 25
            if close[i] > upper_band[i] and volume[i] > 1.5 * vol_ma[i] and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band AND volume > 1.5x average AND 1d ADX > 25
            elif close[i] < lower_band[i] and volume[i] > 1.5 * vol_ma[i] and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below lower band OR ADX weak (< 20)
            if close[i] < lower_band[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above upper band OR ADX weak (< 20)
            if close[i] > upper_band[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals