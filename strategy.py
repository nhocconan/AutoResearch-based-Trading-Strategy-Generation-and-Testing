#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 12h ADX regime filter
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 1d EMA13)
- ADX(14) from 12h: ADX > 25 = trending regime, ADX < 20 = ranging regime
- In trending regime (ADX > 25): trade Elder Ray signals (long when Bull Power > 0, short when Bear Power > 0)
- In ranging regime (ADX < 20): fade extreme Elder Ray (long when Bear Power < -std, short when Bull Power > +std)
- Volume confirmation: current volume > 1.5 * 20-period volume MA
- Discrete signal size: 0.25
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
- Works in both bull/bear: regime adaptation avoids whipsaw in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components
    bull_power = high - ema_13_1d_aligned
    bear_power = ema_13_1d_aligned - low
    
    # Calculate 12h ADX for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            first_avg = np.nansum(data[1:period+1])  # Skip first for TR/DM calc
            result[period] = first_avg
            for i in range(period+1, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 14:
        first_adx = np.nanmean(dx[1:15])  # First 14 DX values after initial
        adx[14] = first_adx
        for i in range(15, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Regime thresholds
    adx_trending = adx_aligned > 25
    adx_ranging = adx_aligned < 20
    
    # Dynamic thresholds for ranging regime (based on 20-period std of Elder Ray)
    bull_power_ma = pd.Series(bull_power).rolling(window=20, min_periods=20).mean().values
    bull_power_std = pd.Series(bull_power).rolling(window=20, min_periods=20).std().values
    bear_power_ma = pd.Series(bear_power).rolling(window=20, min_periods=20).mean().values
    bear_power_std = pd.Series(bear_power).rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need EMA13(1d) + ADX(12h) + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(bull_power_ma[i]) or np.isnan(bull_power_std[i]) or
            np.isnan(bear_power_ma[i]) or np.isnan(bear_power_std[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if adx_trending[i]:
                # Trending regime: follow Elder Ray signals
                if bull_power[i] > 0 and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] > 0 and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            elif adx_ranging[i]:
                # Ranging regime: fade extreme Elder Ray
                if bear_power[i] < -bull_power_std[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                elif bull_power[i] > bear_power_std[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: reverse signal or power deterioration
            if adx_trending[i]:
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif adx_ranging[i]:
                if bear_power[i] > -0.5 * bull_power_std[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or power deterioration
            if adx_trending[i]:
                if bear_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif adx_ranging[i]:
                if bull_power[i] < 0.5 * bear_power_std[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hADX_Regime_v1"
timeframe = "6h"
leverage = 1.0