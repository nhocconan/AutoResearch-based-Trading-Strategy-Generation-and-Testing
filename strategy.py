#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending market)
# Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 (trending market)
# Exit when 1d ADX < 20 (range market) OR power values converge (|Bull Power| + |Bear Power| < threshold)
# Uses 6h primary timeframe with 1d HTF for ADX regime and Elder Ray calculation
# Elder Ray measures bull/bear power relative to EMA13, effective in both bull and bear markets
# ADX regime filter ensures we only trade in trending conditions, avoiding whipsaws in ranging markets
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_1dADX25_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop for ADX regime filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX for regime filter (ADX > 25 = trending)
    if len(df_1d) >= 14:
        # True Range
        tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
        tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
        tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                          (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                          np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
        dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                           (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                          np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+, DM-
        tr_period = 14
        tr_smooth = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        
        tr_smooth[tr_period-1] = np.sum(tr[:tr_period])
        dm_plus_smooth[tr_period-1] = np.sum(dm_plus[:tr_period])
        dm_minus_smooth[tr_period-1] = np.sum(dm_minus[:tr_period])
        
        for i in range(tr_period, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / tr_period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / tr_period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / tr_period) + dm_minus[i]
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.zeros_like(di_plus)
        mask = (di_plus + di_minus) != 0
        dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
        
        adx_period = 14
        adx = np.zeros_like(dx)
        if len(dx) >= 2*adx_period-1:
            adx[2*adx_period-2] = np.sum(dx[adx_period-1:2*adx_period-1]) / adx_period
            for i in range(2*adx_period-1, len(dx)):
                adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    else:
        adx = np.full(len(df_1d), np.nan)
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # We need 6h EMA13 for Elder Ray calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending up)
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 (trending down)
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 1d ADX < 20 (range market) OR power values converge (weakening trend)
            if (adx_aligned[i] < 20 or 
                (abs(bull_power[i]) + abs(bear_power[i])) < 0.001 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 1d ADX < 20 (range market) OR power values converge (weakening trend)
            if (adx_aligned[i] < 20 or 
                (abs(bull_power[i]) + abs(bear_power[i])) < 0.001 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals