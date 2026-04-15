#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h KAMA trend filter and 1d Bollinger Bands for mean reversion entries.
# In 4h KAMA uptrend, wait for price to touch lower 1d Bollinger Band (20,2) for long entry.
# In 4h KAMA downtrend, wait for price to touch upper 1d Bollinger Band for short entry.
# Volume confirmation and session filter (08-20 UTC) reduce false signals.
# Designed for low trade frequency (15-25/year) to minimize fee drag while capturing trend-aligned mean reversion.

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
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: KAMA (adaptive trend filter) ===
    close_4h = df_4h['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.sum(np.abs(np.diff(close_4h, prepend=close_4h[0])), axis=0) if False else np.abs(np.diff(close_4h, prepend=close_4h[0]))  # placeholder
    # Correct ER calculation
    er = np.zeros_like(close_4h)
    for i in range(1, len(close_4h)):
        if i >= 1:
            dir = np.abs(close_4h[i] - close_4h[i-1])
            vol = np.sum(np.abs(np.diff(close_4h[max(0, i-9):i+1]))) if i >= 10 else np.sum(np.abs(np.diff(close_4h[0:i+1])))
            er[i] = dir / (vol + 1e-10) if vol > 0 else 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # === 1d Indicators: Bollinger Bands (20,2) ===
    close_1d = df_1d['close'].values
    basis = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    dev = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = basis + (2 * dev)
    lower_bb = basis - (2 * dev)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 4h KAMA uptrend (price > KAMA)
        # 2. Price touches or crosses below lower 1d Bollinger Band
        # 3. Volume confirmation
        if (close[i] > kama_aligned[i]) and (close[i] <= lower_bb_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. In 4h KAMA downtrend (price < KAMA)
        # 2. Price touches or crosses above upper 1d Bollinger Band
        # 3. Volume confirmation
        elif (close[i] < kama_aligned[i]) and (close[i] >= upper_bb_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_KAMA_BBands20_2_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0