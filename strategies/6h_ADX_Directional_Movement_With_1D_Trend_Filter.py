#!/usr/bin/env python3
# 6h_ADX_Directional_Movement_With_1D_Trend_Filter
# Hypothesis: 6s trend direction from 14-period ADX on 6h chart, filtered by 1d EMA50 trend.
# In bull markets (price > 1d EMA50): long when ADX rising and +DI > -DI.
# In bear markets (price < 1d EMA50): short when ADX rising and -DI > +DI.
# ADX > 25 ensures strong trend, preventing whipsaw in ranging markets.
# Uses Wilder's smoothing for ADX/DI calculation. Entry on bar close to avoid look-ahead.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ADX_Directional_Movement_With_1D_Trend_Filter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ADX and DI on 6h data (Wilder's smoothing)
    period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    atr = np.full_like(high, np.nan)
    dm_plus_smooth = np.full_like(high, np.nan)
    dm_minus_smooth = np.full_like(high, np.nan)
    
    # Initial values (simple average of first 'period' values)
    if len(high) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder's smoothing for subsequent values
        for i in range(period + 1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI and DX
    di_plus = np.full_like(high, np.nan)
    di_minus = np.full_like(high, np.nan)
    dx = np.full_like(high, np.nan)
    
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = (dm_plus_smooth[valid] / atr[valid]) * 100
    di_minus[valid] = (dm_minus_smooth[valid] / atr[valid]) * 100
    
    dx_valid = valid & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])) * 100
    
    # ADX (smoothed DX)
    adx = np.full_like(high, np.nan)
    if len(high) >= 2 * period:
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period + 1, len(high)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2*period + 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1d EMA50
            uptrend = close[i] > ema50_1d_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            # Long: uptrend + ADX > 25 + +DI > -DI
            if uptrend and adx[i] > 25 and di_plus[i] > di_minus[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + ADX > 25 + -DI > +DI
            elif downtrend and adx[i] > 25 and di_minus[i] > di_plus[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend weakens or reverses
            if adx[i] < 20 or di_plus[i] <= di_minus[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend weakens or reverses
            if adx[i] < 20 or di_minus[i] <= di_plus[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals