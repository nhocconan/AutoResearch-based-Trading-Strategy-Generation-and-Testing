#!/usr/bin/env python3
"""
4h_Breakout_Volume_ADX_Filter
Hypothesis: 4-hour price breakouts with volume confirmation and ADX trend filter capture
institutional moves in both bull and bear markets. Uses 4h Donchian breakout (20) for
structure, volume > 1.5x 20-period average for confirmation, and ADX(14) > 25 to filter
for trending conditions, reducing false breakouts in chop. Designed for 4h timeframe to
target 75-200 total trades over 4 years (19-50/year). Stops when price reverses or trend weakens.
"""

name = "4h_Breakout_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX(14) for trend strength filter - calculated on 4h data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values using Wilder's smoothing (EMA-like)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Avoid division by zero
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # 4h data for ADX and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for ADX and Donchian calculation
        return np.zeros(n)
    
    # Calculate ADX on 4h data
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_4h_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and strong trend
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and strong trend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or trend weakens (ADX < 20)
            if (close[i] < donchian_low_aligned[i]) or (adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or trend weakens (ADX < 20)
            if (close[i] > donchian_high_aligned[i]) or (adx_4h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals