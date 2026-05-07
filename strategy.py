#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel (20) AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower channel (20) AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (middle).
# This strategy captures strong trending moves with volatility expansion while avoiding choppy markets.
# The 1d ADX filter ensures we only trade when there is a strong trend (ADX > 25).
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the trend direction only when ADX confirms strength.

name = "12h_DonchianBreakout_1dADX25_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20)
    dc_length = 20
    dc_upper = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    dc_lower = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14)
    adx_length = 14
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Smooth the values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/adx_length, adjust=False).mean().values / \
              pd.Series(tr).ewm(alpha=1/adx_length, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/adx_length, adjust=False).mean().values / \
               pd.Series(tr).ewm(alpha=1/adx_length, adjust=False).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/adx_length, adjust=False).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 30)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(dc_middle[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, ADX > 25, volume filter
            long_cond = (close[i] > dc_upper[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, ADX > 25, volume filter
            short_cond = (close[i] < dc_lower[i]) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian channel (below middle)
            if close[i] < dc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian channel (above middle)
            if close[i] > dc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals