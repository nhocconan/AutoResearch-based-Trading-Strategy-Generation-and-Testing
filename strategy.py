#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1-week EMA200 trend filter and volume confirmation.
# Uses Donchian(20) breakouts on daily timeframe for fewer, higher-quality trades.
# 1-week EMA200 filters for trend direction to avoid counter-trend entries.
# Volume > 1.5x 20-period EMA ensures institutional participation.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
name = "1d_Donchian20_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period rolling max/min for Donchian channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1-week EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to be valid
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spill and above 1w EMA200
            if (price > high_20[i] and vol_spike[i] and price > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume spill and below 1w EMA200
            elif (price < low_20[i] and vol_spike[i] and price < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Donchian lower (mean reversion)
            if price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above Donchian upper (mean reversion)
            if price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals