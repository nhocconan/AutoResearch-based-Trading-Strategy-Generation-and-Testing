#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day volume spike and ADX trend filter.
Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period average and ADX > 25.
Short when price breaks below 20-period Donchian low with volume > 1.5x 20-period average and ADX > 25.
Exit when price crosses the 10-period EMA.
Designed for low trade frequency (~20-40/year) to avoid fee drag while capturing trends.
Works in trending markets by filtering weak breakouts and sideways markets via ADX.
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
    
    # Load 1-day data for volume filter and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1-day ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(val, period):
        result = np.full_like(val, np.nan, dtype=float)
        if len(val) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(val[:period])
        # Wilder's smoothing
        for i in range(period, len(val)):
            result[i] = (result[i-1] * (period-1) + val[i]) / period
        return result
    
    atr = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr, np.nan, dtype=float)
    di_minus = np.full_like(atr, np.nan, dtype=float)
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = (dm_plus_smooth[valid] / atr[valid]) * 100
    di_minus[valid] = (dm_minus_smooth[valid] / atr[valid]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    di_sum = di_plus + di_minus
    valid_dx = ~np.isnan(di_sum) & (di_sum != 0)
    dx[valid_dx] = (np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]) * 100
    
    adx = smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 4h data
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and trend confirmation
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma_1d_aligned[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and trend confirmation
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_ma_1d_aligned[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses 10-period EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below EMA
                if close[i] < ema_10[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above EMA
                if close[i] > ema_10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0