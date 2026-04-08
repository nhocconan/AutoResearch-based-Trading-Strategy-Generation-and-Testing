#!/usr/bin/env python3
# 6h_1d_elder_ray_regime_v1
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX>25) to trade with higher timeframe momentum.
# Long: Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 AND 1d +DI > -DI (bullish regime)
# Short: Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 AND 1d -DI > +DI (bearish regime)
# Exit: Elder Ray divergence (Bull Power < 0 for long, Bear Power > 0 for short) OR ADX < 20 (regime change)
# Uses 6h primary timeframe with 1d HTF for ADX/DI regime filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for ADX/DI regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX and DI (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Rest: EMA
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI and ADX
    plus_di14 = np.where(tr14 > 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 > 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) > 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx14 = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    adx14_aligned = align_htf_to_ltf(prices, df_1d, adx14)
    plus_di14_aligned = align_htf_to_ltf(prices, df_1d, plus_di14)
    minus_di14_aligned = align_htf_to_ltf(prices, df_1d, minus_di14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx14_aligned[i]) or np.isnan(plus_di14_aligned[i]) or np.isnan(minus_di14_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        adx = adx14_aligned[i]
        pdi = plus_di14_aligned[i]
        mdi = minus_di14_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR ADX < 20 (regime change)
            if bp < 0 or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR ADX < 20 (regime change)
            if br > 0 or adx < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND +DI > -DI
            if bp > 0 and br < 0 and adx > 25 and pdi > mdi:
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 AND -DI > +DI
            elif br < 0 and bp > 0 and adx > 25 and mdi > pdi:
                position = -1
                signals[i] = -0.25
    
    return signals