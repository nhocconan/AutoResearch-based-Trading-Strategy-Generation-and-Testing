#!/usr/bin/env python3
"""
Hypothesis: 1d strategy combining Donchian breakout with 1w ADX trend filter and volume confirmation.
Donchian(20) breakouts capture momentum. ADX(14) > 25 on weekly ensures trending market.
Volume spike (>1.5x 20-day average) confirms breakout strength. Works in both bull and bear
markets by filtering for trending conditions only. Target: 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    # Weekly ADX aligned to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Daily Donchian(20)
    high_max = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Daily volume confirmation
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + ADX > 25 + volume spike
            if (price_close > high_max[i] and 
                adx_val > 25 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + ADX > 25 + volume spike
            elif (price_close < low_min[i] and 
                  adx_val > 25 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend weakening (ADX < 20) or opposite Donchian break
            if position == 1:
                if adx_val < 20 or price_close < low_min[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if adx_val < 20 or price_close > high_max[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_1wADX_Volume"
timeframe = "1d"
leverage = 1.0