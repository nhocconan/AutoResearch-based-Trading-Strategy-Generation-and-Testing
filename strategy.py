#!/usr/bin/env python3
"""
4h_KAMA_Direction_VolumeSpike_ChopFilter_v1
Hypothesis: 4h KAMA (adaptive trend) direction filter with volume spike (>2x avg volume) and choppiness regime (CHOP > 61.8 = ranging) for mean reversion entries. Long when KAMA rising + volume spike + chop > 61.8 + price < KAMA (oversold in range). Short when KAMA falling + volume spike + chop > 61.8 + price > KAMA (overbought in range). Uses discrete sizing 0.25 to minimize fee churn. Works in sideways markets by fading extremes within the range, avoiding strong trends where chop < 38.2.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for KAMA, volume, CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA Calculation (adaptive trend) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum |close[i]-close[i-1]| over 10
    # Pad arrays to align with close
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- Volume Spike Confirmation ---
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- Choppiness Index (CHOP) ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.where(range_hl != 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA, 20 for volume, 14 for CHOP)
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        kama_val = kama[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        chop_val = chop[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(avg_vol) or np.isnan(chop_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        # Chop filter: CHOP > 61.8 = ranging market (mean reversion zone)
        chop_ranging = chop_val > 61.8
        
        # Long logic: KAMA rising + volume spike + chop ranging + price < KAMA (oversold)
        kama_rising = kama_val > kama[i-1] if i > 0 else False
        long_condition = kama_rising and volume_confirmed and chop_ranging and (close_val < kama_val)
        # Short logic: KAMA falling + volume spike + chop ranging + price > KAMA (overbought)
        kama_falling = kama_val < kama[i-1] if i > 0 else False
        short_condition = kama_falling and volume_confirmed and chop_ranging and (close_val > kama_val)
        
        # Exit logic: reverse conditions or chop exits ranging (trend start)
        long_exit = (position == 1 and (not kama_rising or not chop_ranging or close_val >= kama_val))
        short_exit = (position == -1 and (not kama_falling or not chop_ranging or close_val <= kama_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_KAMA_Direction_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0