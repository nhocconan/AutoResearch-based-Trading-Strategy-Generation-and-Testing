#!/usr/bin/env python3
"""
6h_Range_Filtered_Breakout_v1
Hypothesis: Use 12h Donchian breakout with 12h ADX trend filter and 6h volume confirmation.
Works in trending markets (ADX > 25) and avoids false breakouts in ranging markets.
Targets 12-30 trades/year to minimize fee drag.
"""

name = "6h_Range_Filtered_Breakout_v1"
timeframe = "6h"
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
    
    # 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower (20-period high/low)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # Avoid division by zero
        dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
        dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
        
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX
        dx = np.zeros_like(di_plus)
        denom = di_plus + di_minus
        dx = np.where(denom != 0, 100 * np.abs(di_plus - di_minus) / denom, 0)
        
        # ADX = smoothed DX
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, df_12h['close'].values, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND ADX > 25 (trending) with volume confirmation
            if close[i] > donch_high[i] and adx_12h_aligned[i] > 25 and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND ADX > 25 (trending) with volume confirmation
            elif close[i] < donch_low[i] and adx_12h_aligned[i] > 25 and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below Donchian low (mean reversion or trend end)
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above Donchian high (mean reversion or trend end)
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals