#!/usr/bin/env python3
# 12H_ADX_TREND_STRENGTH_WITH_VOLUME_CONFIRMATION
# Hypothesis: Use ADX to identify strong trends on daily timeframe, combined with volume confirmation.
# In strong uptrend (ADX > 25 and +DI > -DI), go long when volume spikes above average.
# In strong downtrend (ADX > 25 and -DI > +DI), go short when volume spikes above average.
# Uses volume confirmation to avoid false breakouts and reduce whipsaws.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_ADX_TREND_STRENGTH_WITH_VOLUME_CONFIRMATION"
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
    
    # Daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # Avoid division by zero
    dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
    dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(di_plus)
    mask = (di_plus + di_minus) != 0
    dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
    
    # ADX is smoothed DX
    adx = wilders_smoothing(dx, period)
    
    # Align to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # Volume indicators on 12h timeframe
    vol_ma = np.zeros(n)
    vol_std = np.zeros(n)
    vol_threshold = np.zeros(n)
    
    # Calculate volume moving average and standard deviation
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
        vol_std[i] = np.std(volume[i-20:i])
        vol_threshold[i] = vol_ma[i] + 2.0 * vol_std[i]  # Volume spike threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or 
            np.isnan(di_minus_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Strong uptrend (ADX > 25 and +DI > -DI) + volume spike
            if (adx_aligned[i] > 25 and 
                di_plus_aligned[i] > di_minus_aligned[i] and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Strong downtrend (ADX > 25 and -DI > +DI) + volume spike
            elif (adx_aligned[i] > 25 and 
                  di_minus_aligned[i] > di_plus_aligned[i] and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakening or reversal
            if (adx_aligned[i] < 20 or 
                di_plus_aligned[i] < di_minus_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakening or reversal
            if (adx_aligned[i] < 20 or 
                di_minus_aligned[i] < di_plus_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals