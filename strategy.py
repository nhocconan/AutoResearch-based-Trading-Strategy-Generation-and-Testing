#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with volume confirmation and weekly pivot direction filter
# Long when price breaks above 6h Donchian(20) high + volume > 1.5x 20-period avg + price above weekly pivot point
# Short when price breaks below 6h Donchian(20) low + volume > 1.5x 20-period avg + price below weekly pivot point
# Weekly pivot acts as regime filter: above = bullish bias, below = bearish bias
# Designed for low trade frequency (12-25/year) to minimize fee drag in bear markets (2025+ test)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h HTF data once before loop (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 40:
        return np.zeros(n)
    
    # === 6h Indicator: Donchian Channel (20) ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    highest_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, lowest_low_20)
    
    # Get 1w HTF data once before loop (weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h Donchian(20) high
        # 2. Above weekly pivot (bullish regime)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i]) and \
           (close[i] > weekly_pivot_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h Donchian(20) low
        # 2. Below weekly pivot (bearish regime)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i]) and \
             (close[i] < weekly_pivot_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_Volume_WeeklyPivot_Filter_v1"
timeframe = "6h"
leverage = 1.0