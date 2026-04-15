#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian breakout for trend direction and 1d Elder Ray (Bull/Bear Power) for entry timing.
# In 12h uptrend (price > upper Donchian), go long when 1d Bull Power turns positive (buying pressure emerging).
# In 12h downtrend (price < lower Donchian), go short when 1d Bear Power turns negative (selling pressure emerging).
# Volume confirmation ensures institutional participation. Target: 12-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian Channel (20) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # === 1d Indicators: Elder Ray (Bull Power / Bear Power) ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (reduces noise, maintains edge)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 12h uptrend (price > 12h upper Donchian)
        # 2. 1d Bull Power > 0 (buying pressure > EMA13)
        # 3. Volume confirmation
        if (close[i] > donchian_high_12h_aligned[i]) and (bull_power_1d_aligned[i] > 0) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 12h downtrend (price < 12h lower Donchian)
        # 2. 1d Bear Power < 0 (selling pressure > EMA13)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_12h_aligned[i]) and (bear_power_1d_aligned[i] < 0) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1dElderRay_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0