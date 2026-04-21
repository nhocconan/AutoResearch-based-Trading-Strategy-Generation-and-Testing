#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above Donchian upper with volume > 1.5x average and daily ADX > 25.
Short when price breaks below Donchian lower with volume > 1.5x average and daily ADX > 25.
Exit when price returns to Donchian middle or volume drops below average.
Donchian channels provide clear breakout levels, volume confirms breakout strength,
and ADX ensures we only trade in trending markets to avoid whipsaws in chop.
Target: 20-30 trades/year for low fee drag and robust performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
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
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = WilderSmooth(tr, 14)
    dm_plus_smooth = WilderSmooth(dm_plus, 14)
    dm_minus_smooth = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    adx = np.concatenate([np.full(27, np.nan), adx[27:]])  # First 27 values NaN
    
    # Align daily ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    donchian_up = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_down = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_up + donchian_down) / 2
    
    # Calculate 4h volume average (20-period)
    volume_4h = prices['volume'].values
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_down[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_current = volume_4h[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper, volume surge, daily ADX > 25 (trending)
            if (price_close > donchian_up[i] and 
                vol_current > 1.5 * vol_ma_20[i] and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower, volume surge, daily ADX > 25 (trending)
            elif (price_close < donchian_down[i] and 
                  vol_current > 1.5 * vol_ma_20[i] and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to Donchian middle or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= middle or volume < average
                if (price_close <= donchian_middle[i] or
                    vol_current < vol_ma_20[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= middle or volume < average
                if (price_close >= donchian_middle[i] or
                    vol_current < vol_ma_20[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume1.5x_ADX25"
timeframe = "4h"
leverage = 1.0