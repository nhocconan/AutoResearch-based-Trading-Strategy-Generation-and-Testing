#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation.
# Uses 1w EMA200 for long-term trend bias and Donchian channel breakouts on 6h for entry timing.
# Includes volume filter (current volume > 1.8x 20-bar 6h volume SMA) to avoid low-momentum breakouts.
# Designed for low trade frequency (12-30/year) to minimize fee drag. Works in bull/bear:
# - Bull: long when price breaks above Donchian upper AND above 1w EMA200
# - Bear: short when price breaks below Donchian lower AND below 1w EMA200
# Volume confirmation ensures breakouts have conviction, reducing false signals in chop.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h and 1w HTF data once before loop
    df_6h = get_htf_data(prices, '6h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_6h) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 6h Indicators: Donchian Channel (20) ===
    high_6h = pd.Series(df_6h['high'].values)
    low_6h = pd.Series(df_6h['low'].values)
    donchian_upper = high_6h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_6h.rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(200) for long-term trend bias
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 6h volume > 1.8x 20-period 6h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper channel (breakout)
        # 2. Price above 1w EMA200 (bullish long-term trend)
        # 3. Volume confirmation
        if (close[i] > donchian_upper_aligned[i] and
            close[i] > ema_200_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower channel (breakdown)
        # 2. Price below 1w EMA200 (bearish long-term trend)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower_aligned[i] and
              close[i] < ema_200_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1wEMA200_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0