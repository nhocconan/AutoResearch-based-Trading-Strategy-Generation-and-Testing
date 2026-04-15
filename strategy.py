#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Donchian channel breakouts with 1w EMA200 trend filter and volume confirmation.
# In trending markets (price above/below 1w EMA200), trade breakouts of 1d Donchian(20) in trend direction.
# In ranging markets (price near 1w EMA200), fade Donchian extremes for mean reversion.
# Volume filter ensures momentum validity. Designed for low trade frequency (12-30/year) to minimize fee drag.
# Works in both bull and bear markets via trend-adaptive logic.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min for Donchian channels
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(200) for long-term trend bias
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Trending market: price outside 1w EMA200 bands (using 10% deviation)
        # Ranging market: price near 1w EMA200 (within 10% deviation)
        
        ema_val = ema_200_1w_aligned[i]
        upper_band = ema_val * 1.10
        lower_band = ema_val * 0.90
        
        in_uptrend = close[i] > upper_band
        in_downtrend = close[i] < lower_band
        in_range = (lower_band <= close[i] <= upper_band)
        
        # === LONG CONDITIONS ===
        # 1. In uptrend AND breakout above 1d Donchian high (continuation)
        # 2. In ranging market AND price at 1d Donchian low (mean reversion long)
        if vol_confirm:
            if (in_uptrend and close[i] > donchian_high_aligned[i]) or \
               (in_range and close[i] <= donchian_low_aligned[i] * 1.001):
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In downtrend AND breakdown below 1d Donchian low (continuation)
        # 2. In ranging market AND price at 1d Donchian high (mean reversion short)
        elif vol_confirm:
            if (in_downtrend and close[i] < donchian_low_aligned[i]) or \
               (in_range and close[i] >= donchian_high_aligned[i] * 0.999):
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1wEMA200_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0