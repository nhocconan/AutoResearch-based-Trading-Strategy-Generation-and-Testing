#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Uses 1d EMA(50) for trend bias, 12h Camarilla levels for entry, and volume > 2.0x 20-bar SMA for confirmation.
# Designed for low trade frequency (12-37/year) to minimize fee drag. Works in bull/bear: EMA50 avoids counter-trend trades,
# Camarilla breakouts capture momentum with institutional relevance. Volume filter ensures breakouts have conviction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Calculate from previous 12h bar's OHLC
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    close_12h = pd.Series(df_12h['close'].values)
    
    # Pivot point and Camarilla levels
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r1 = pivot + (range_12h * 1.1 / 12)
    s1 = pivot - (range_12h * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1.values)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1
        # 2. 1d price above EMA50 (bullish trend bias)
        # 3. Volume confirmation (>2.0x average)
        if (close[i] > r1_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1
        # 2. 1d price below EMA50 (bearish trend bias)
        # 3. Volume confirmation (>2.0x average)
        elif (close[i] < s1_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_EMA50_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0