#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R1/S1 for mean reversion, R2/S2 for breakout)
# combined with 1w EMA200 trend filter and volume confirmation. In ranging markets (price between R1-S1),
# fade extremes; in trending markets (price outside R2-S2), breakout continuation. Volume filter ensures
# momentum validity. Designed for low trade frequency (12-30/year) to minimize fee drag while adapting
# to regime via pivot structure. Works in both bull and bear via regime-adaptive logic.

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
    
    # === 1d Indicators: Camarilla Pivot Levels (using typical price) ===
    # Typical price = (high + low + close) / 3
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Calculate pivot and support/resistance levels
    pivot_1d = typical_price_1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    r2_1d = pivot_1d + (range_1d * 1.1 / 6)
    s2_1d = pivot_1d - (range_1d * 1.1 / 6)
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # === 1w Indicators: Trend Filter ===
    # 1w EMA(200) for long-term trend bias
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
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
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Ranging market: price between R1 and S1
        # Trending market: price outside R2 and S2
        # Transition zone: between R1-S1 and R2-S2 (no trade)
        
        in_range = (s1_aligned[i] <= close[i] <= r1_aligned[i])
        in_uptrend = close[i] > r2_aligned[i]
        in_downtrend = close[i] < s2_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. In ranging market AND price at S1 support (mean reversion long)
        # 2. OR in uptrend AND breakout above R2 (continuation long)
        # 3. Volume confirmation
        if vol_confirm:
            if (in_range and close[i] <= s1_aligned[i] * 1.001) or \
               (in_uptrend and close[i] > r2_aligned[i]):
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In ranging market AND price at R1 resistance (mean reversion short)
        # 2. OR in downtrend AND breakdown below S2 (continuation short)
        # 3. Volume confirmation
        elif vol_confirm:
            if (in_range and close[i] >= r1_aligned[i] * 0.999) or \
               (in_downtrend and close[i] < s2_aligned[i]):
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_R2S2_EMA200_VolFilter_v1"
timeframe = "12h"
leverage = 1.0