#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: 4h TRIX (12,20,9) with volume spike (>2x average) and chop regime filter (CHOP(14) < 38.2 for trending). 
TRIX captures momentum with reduced lag, volume confirms breakout strength, chop filter ensures we only trade in trending markets.
Works in both bull and bear markets by following TRIX direction - long when TRIX rising, short when falling.
Discrete position sizing (0.25) minimizes fee churn. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for TRIX and CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX: EMA(EMA(EMA(close,12),12),12) then ROC
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (pd.Series(ema3).pct_change().values)
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values  # Signal line
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / log10((HHV(14)-LLV(14)) * sqrt(14)))
    # Simplified: CHOP = 100 * log10(ATR_sum / (range * sqrt(period))) / log10(sqrt(period))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high).rolling(window=14, min_periods=14).max().values
    llvl = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = hhvl - llvl
    sqrt14 = np.sqrt(14)
    chop = 100 * (np.log10(atr_sum / (range_hl * sqrt14 + 1e-10)) / np.log10(sqrt14))
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 40 for TRIX, 14 for CHOP, 20 for volume)
    start_idx = max(40, 14, 20)
    
    for i in range(start_idx, n):
        # Get current values
        trix_val = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        chop_val = chop[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Skip if any data not ready
        if (np.isnan(trix_val) or np.isnan(trix_prev) or np.isnan(chop_val) or 
            np.isnan(avg_vol)):
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
        
        # Chop regime filter: CHOP < 38.2 = trending market (good for momentum)
        trending_regime = chop_val < 38.2
        
        # TRIX rising = bullish momentum, TRIX falling = bearish momentum
        trix_rising = trix_val > trix_prev
        trix_falling = trix_val < trix_prev
        
        # Long logic: TRIX rising + volume confirmation + trending regime
        long_condition = trix_rising and volume_confirmed and trending_regime
        # Short logic: TRIX falling + volume confirmation + trending regime
        short_condition = trix_falling and volume_confirmed and trending_regime
        
        # Exit logic: TRIX momentum reversal
        exit_long = trix_falling  # TRIX starts falling
        exit_short = trix_rising  # TRIX starts rising
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0