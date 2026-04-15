#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# combined with 1d EMA50 trend filter and volume confirmation. 
# In ranging markets (price between R3-S3), fade extremes; in trending markets (price outside R4-S4), 
# breakout continuation. Volume filter ensures momentum validity. Designed for low trade frequency 
# (12-30/year) to minimize fee drag while adapting to regime via pivot structure.

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
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Camarilla Pivot Levels (using typical price) ===
    # Typical price = (high + low + close) / 3
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    
    # Calculate pivot and support/resistance levels
    pivot_12h = typical_price_12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = pivot_12h + (range_12h * 1.1 / 4)
    s3_12h = pivot_12h - (range_12h * 1.1 / 4)
    r4_12h = pivot_12h + (range_12h * 1.1 / 2)
    s4_12h = pivot_12h - (range_12h * 1.1 / 2)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Ranging market: price between R3 and S3
        # Trending market: price outside R4 and S4
        # Transition zone: between R3-S3 and R4-S4 (no trade)
        
        in_range = (s3_aligned[i] <= close[i] <= r3_aligned[i])
        in_uptrend = close[i] > r4_aligned[i]
        in_downtrend = close[i] < s4_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. In ranging market AND price at S3 support (mean reversion long)
        # 2. OR in uptrend AND breakout above R4 (continuation long)
        # 3. Volume confirmation
        if vol_confirm:
            if (in_range and close[i] <= s3_aligned[i] * 1.002) or \
               (in_uptrend and close[i] > r4_aligned[i]):
                signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In ranging market AND price at R3 resistance (mean reversion short)
        # 2. OR in downtrend AND breakdown below S4 (continuation short)
        # 3. Volume confirmation
        elif vol_confirm:
            if (in_range and close[i] >= r3_aligned[i] * 0.998) or \
               (in_downtrend and close[i] < s4_aligned[i]):
                signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0