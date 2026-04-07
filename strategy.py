#!/usr/bin/env python3
"""
6h Williams Alligator with 1d ADX Trend Filter
Long when Alligator jaws < teeth < lips and 1d ADX > 25 (trending up)
Short when Alligator jaws > teeth > lips and 1d ADX > 25 (trending down)
Exit when Alligator alignment breaks or ADX < 20
Designed to capture strong trends in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_alligator_1d_adx_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Williams Alligator ===
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            # First value is SMA
            result[period-1] = np.mean(data[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_PRICE) / N
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_price := (high + low) / 2, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift jaws, teeth, lips as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # First few values become NaN due to roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # === 1d ADX Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum.reduce([tr1, tr2, tr3])])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            # First value is average of first 'period' values
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CURRENT_VALUE) / N
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period)
    
    # Align to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Wait for sufficient data
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (jaws > teeth) OR ADX < 20 (trend weak)
            if jaw[i] > teeth[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (jaws < teeth) OR ADX < 20 (trend weak)
            if jaw[i] < teeth[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Strong trend: ADX > 25
            if adx_aligned[i] > 25:
                # Perfect alignment: jaws < teeth < lips (bullish)
                if jaw[i] < teeth[i] and teeth[i] < lips[i]:
                    position = 1
                    signals[i] = 0.25
                # Perfect alignment: jaws > teeth > lips (bearish)
                elif jaw[i] > teeth[i] and teeth[i] > lips[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals