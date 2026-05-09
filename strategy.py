#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX filter.
# In bull markets: breakouts capture momentum. In bear markets: ADX>25 filters whipsaws,
# and volume confirmation ensures breakouts are institutional. Low trade frequency (~25/year)
# avoids fee drag. Uses discrete positions (0.0, ±0.25) to minimize churn.

name = "4h_Donchian20_Volume_ADX1d"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period ADX on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(x, period):
        result = np.zeros_like(x)
        result[period-1] = np.nansum(x[:period])
        for i in range(period, len(x)):
            result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus14 = wilders_smoothing(dm_plus, 14)
    dm_minus14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # 4h Donchian(20) channels
    def donchian_channels(h, l, period):
        upper = np.zeros_like(h)
        lower = np.zeros_like(l)
        for i in range(len(h)):
            if i >= period - 1:
                upper[i] = np.max(h[i-(period-1):i+1])
                lower[i] = np.min(l[i-(period-1):i+1])
            else:
                upper[i] = np.nan
                lower[i] = np.nan
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Volume filter: 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma[i] = np.nan
    
    # Align 1d ADX to 4h
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > vol_ma[i] * 1.5
        adx_filter = adx_4h[i] > 25  # Only trade when trending
        
        if position == 0:
            # Long: break above upper Donchian with volume and ADX filter
            if close[i] > upper[i] and vol_ok and adx_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume and ADX filter
            elif close[i] < lower[i] and vol_ok and adx_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower Donchian
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper Donchian
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals