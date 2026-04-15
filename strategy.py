#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla pivot R1/S1 breakout with volume confirmation and 1d chop regime filter
# Long when price breaks above 1w Camarilla R1 + volume > 1.5x 20-period avg + 1d chop < 61.8 (trending)
# Short when price breaks below 1w Camarilla S1 + volume > 1.5x 20-period avg + 1d chop < 61.8 (trending)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Camarilla pivots provide mathematically derived support/resistance. Chop filter ensures we only trade trending markets.
# Works in bull markets (breakout continuation) and bear markets (strong downtrend continuation) by requiring trending regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1w Indicator: Camarilla Pivot Levels (R1, S1) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r1 = pivot + (range_1w * 1.0 / 12.0)  # R1
    s1 = pivot - (range_1w * 1.0 / 12.0)  # S1
    
    # Align to LTF (12h)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get 1d HTF data once before loop for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Choppiness Index (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period) using Wilder's smoothing
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of TR over atr_period
    tr_sum = np.zeros_like(tr)
    tr_sum[atr_period-1] = np.sum(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - tr[i-atr_period] + tr[i]
    
    # Choppiness Index: 100 * log10(tr_sum / (atr * atr_period)) / log10(atr_period)
    # Avoid division by zero and log of zero
    chop = np.full_like(tr, 50.0)  # default to neutral
    mask = (atr > 0) & (tr_sum > 0) & (atr_period > 0)
    if np.any(mask):
        chop[mask] = 100 * np.log10(tr_sum[mask] / (atr[mask] * atr_period)) / np.log10(atr_period)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(atr_period*2, 20) + 20  # chop(14) + volume(20) + extra
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Chop filter: trending market (chop < 61.8)
        chop_filter = chop_aligned[i] < 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1w Camarilla R1
        # 2. Volume confirmation
        # 3. Trending regime (chop < 61.8)
        if (close[i] > r1_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1w Camarilla S1
        # 2. Volume confirmation
        # 3. Trending regime (chop < 61.8)
        elif (close[i] < s1_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1w_1dChop_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0