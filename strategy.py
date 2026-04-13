#!/usr/bin/env python3
"""
4h_12h_Donchian_Breakout_Volume_Confirmation_v1
Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ADX trend filter.
Long when price breaks above 4h upper Donchian band + 12h volume > 1.5x 20-period average + ADX > 25.
Short when price breaks below 4h lower Donchian band + 12h volume > 1.5x 20-period average + ADX > 25.
Exit when price crosses the Donchian middle (20-period mean) or ADX < 20 (trend weakness).
Designed for 4h timeframe to target 20-40 trades/year with strong trend capture in both bull/bear markets.
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
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_mid = (high_20 + low_20) / 2
    
    # Get 12h data for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h volume average
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20.values)
    
    # 12h ADX calculation (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- (14-period Wilder's smoothing)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: smoothed = prev - (prev/period) + current
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    tr_14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align 12h indicators to 4h
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20.values)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x 20-period average
        # Get current 12h volume (need to align it)
        idx_12h = i // (12*4)  # 12h = 48 * 15m bars, but we're on 4h = 16 * 15m bars
        # Simpler: use aligned volume data from df_12h
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
        vol_condition = vol_12h_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # ADX condition: trending market
        adx_condition = adx_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > high_20[i]
        short_breakout = close[i] < low_20[i]
        
        # Exit conditions
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        trend_weak = adx_aligned[i] < 20
        
        if position == 0:
            if long_breakout and vol_condition and adx_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and adx_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit or trend_weak:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit or trend_weak:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0