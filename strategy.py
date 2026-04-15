#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel (20) breakout with 1w EMA50 trend filter
# and volume confirmation. In ranging markets (price inside Donchian channel), 
# fade at channel edges; in trending markets (price outside 1.5x Donchian width), 
# breakout continuation. Volume filter ensures momentum validity. Designed for 
# low trade frequency (12-30/year) to minimize fee drag while adapting to regime 
# via Donchian structure. Works in both bull and bear via regime detection.

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
    
    # Donchian high/low (20-period)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_high_1d + donch_low_1d) / 2.0
    donch_width_1d = donch_high_1d - donch_low_1d
    
    # Align to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    donch_width_aligned = align_htf_to_ltf(prices, df_1d, donch_width_1d)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(50) for trend bias
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
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
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or np.isnan(donch_width_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Ranging market: price inside Donchian channel
        # Strong trending market: price outside 1.5x Donchian width from midpoint
        # Weak trend/transition: between Donchian and 1.5x width (no trade)
        
        in_range = (donch_low_aligned[i] <= close[i] <= donch_high_aligned[i])
        strong_uptrend = close[i] > (donch_mid_aligned[i] + 1.5 * donch_width_aligned[i])
        strong_downtrend = close[i] < (donch_mid_aligned[i] - 1.5 * donch_width_aligned[i])
        
        # === LONG CONDITIONS ===
        # 1. In ranging market AND price at Donchian low (mean reversion long)
        # 2. OR in strong uptrend AND breakout above Donchian high (continuation long)
        # 3. Volume confirmation
        if vol_confirm:
            if (in_range and close[i] <= donch_low_aligned[i] * 1.001) or \
               (strong_uptrend and close[i] > donch_high_aligned[i]):
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In ranging market AND price at Donchian high (mean reversion short)
        # 2. OR in strong downtrend AND breakdown below Donchian low (continuation short)
        # 3. Volume confirmation
        elif vol_confirm:
            if (in_range and close[i] >= donch_high_aligned[i] * 0.999) or \
               (strong_downtrend and close[i] < donch_low_aligned[i]):
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_EMA50_1w_VolFilter_v1"
timeframe = "12h"
leverage = 1.0