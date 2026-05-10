#!/usr/bin/env python3
# 6h_MultiTimeframe_CCI_Divergence
# Hypothesis: Combines 6h CCI (20) for overbought/oversold signals with 1d ADX (14) for trend strength.
# Long when CCI < -100 (oversold) and 1d ADX > 25 (trending up); short when CCI > 100 (overbought) and 1d ADX > 25 (trending down).
# Uses volume confirmation (current bar volume > 20-period average) to avoid false signals.
# Designed for 6h timeframe to capture medium-term reversals in both bull and bear markets.
# Target: 20-40 trades/year per symbol.

name = "6h_MultiTimeframe_CCI_Divergence"
timeframe = "6h"
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
    
    # CCI (20) calculation
    typical_price = (high + low + close) / 3.0
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    cci = np.divide(typical_price - tp_mean, 0.015 * tp_std, out=np.full_like(typical_price, np.nan), where=tp_std!=0)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma
    
    # 1d ADX (14) calculation for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.divide(dm_plus_smooth, atr, out=np.full_like(dm_plus_smooth, np.nan), where=atr!=0) * 100
    di_minus = np.divide(dm_minus_smooth, atr, out=np.full_like(dm_minus_smooth, np.nan), where=atr!=0) * 100
    
    # DX and ADX
    dx = np.divide(np.abs(di_plus - di_minus), (di_plus + di_minus), out=np.full_like(di_plus, np.nan), where=(di_plus + di_minus)!=0) * 100
    adx = np.full_like(dx, np.nan)
    # ADX is smoothed DX
    for i in range(14, len(dx)):
        if np.isnan(adx[i-1]):
            adx[i] = np.nanmean(dx[i-13:i+1])  # First ADX is average of first 14 DX
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14  # Wilder smoothing
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(cci[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CCI oversold (< -100), ADX > 25 (trending up), volume confirmation
            if (cci[i] < -100 and
                adx_aligned[i] > 25 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: CCI overbought (> 100), ADX > 25 (trending down), volume confirmation
            elif (cci[i] > 100 and
                  adx_aligned[i] > 25 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: CCI crosses above -100 (leaving oversold) or ADX drops below 20 (weakening trend)
            if (cci[i] > -100 or
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: CCI crosses below 100 (leaving overbought) or ADX drops below 20
            if (cci[i] < 100 or
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals