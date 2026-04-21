#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_TrendFilter
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume spike and daily trend filter (price above/below daily EMA50) yield high-probability trades. Works in bull/bear markets by only taking breakouts in direction of daily trend. Limited entries via volume confirmation and trend alignment reduce overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 4h close
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if daily EMA not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > daily EMA50 for long, price < daily EMA50 for short
        trend_long = price > ema_50_1d_aligned[i]
        trend_short = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + uptrend
            if price > upper[i] and volume_ok and trend_long:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + downtrend
            elif price < lower[i] and volume_ok and trend_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend turns bearish
            if price < lower[i] or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend turns bullish
            if price > upper[i] or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0