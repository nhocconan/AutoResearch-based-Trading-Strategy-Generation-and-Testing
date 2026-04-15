#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian channel breakout with volume confirmation and choppiness regime filter.
# In 1w uptrend (price > upper Donchian(20)), go long when 12h volume > 2.0x 24-period volume SMA and chop < 61.8 (trending).
# In 1w downtrend (price < lower Donchian(20)), go short when 12h volume > 2.0x 24-period volume SMA and chop < 61.8.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-37/year) to overcome fee drag
# and capture sustained trends in both bull and bear markets via HTF structure confirmation.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w Indicators: Donchian Channel (20) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === 12h Indicators: Choppiness Index (14) and Volume SMA (24) ===
    # Choppiness Index: 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low))) over period
    # Simplified: using ATR and range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high - lowest_low
    
    # Avoid division by zero
    sum_atr = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(sum_atr / (range_14 * np.log10(14) + 1e-10))
    chop = np.where(range_14 > 0, chop_raw, 50.0)  # default to neutral when range is zero
    
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 1w uptrend (price > 1w upper Donchian)
        # 2. Volume confirmation
        # 3. Chop < 61.8 (trending regime, not choppy)
        if (close[i] > donchian_high_aligned[i]) and vol_confirm and (chop[i] < 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 1w downtrend (price < 1w lower Donchian)
        # 2. Volume confirmation
        # 3. Chop < 61.8 (trending regime, not choppy)
        elif (close[i] < donchian_low_aligned[i]) and vol_confirm and (chop[i] < 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_1wDonchian20_VolumeChopFilter_v1"
timeframe = "12h"
leverage = 1.0