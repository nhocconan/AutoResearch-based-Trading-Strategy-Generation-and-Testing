#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: On 4h timeframe, enter long when TRIX crosses above zero AND volume > 2.0x 20-period average AND choppiness index > 61.8 (ranging market). Enter short when TRIX crosses below zero AND volume spike AND chop > 61.8. Uses TRIX (TRIple Exponential Average) for smooth momentum, volume confirmation to avoid false signals, and chop regime filter to trade mean reversion in ranging markets (works in both bull and bear). Designed for low-moderate trade frequency (15-30/year) with strong edge in sideways markets where BTC/ETH often consolidate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period ago
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix_values = trix.values
    trix_prev = np.roll(trix_values, 1)
    trix_prev[0] = 0
    trix_cross_up = (trix_values > 0) & (trix_prev <= 0)
    trix_cross_down = (trix_values < 0) & (trix_prev >= 0)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Choppiness Index (14-period) - measures ranging vs trending
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(n))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=1, min_periods=1).sum()  # ATR(1) = TR
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.maximum(max_high - min_low, 1e-10)) / np.log10(14)
    chop_values = chop.values
    chop_high = chop_values > 61.8  # ranging market (mean reversion regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX warmup (45), volume MA warmup (20), chop warmup (14)
    start_idx = max(45, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_values[i]) or np.isnan(volume_ma[i]) or np.isnan(chop_values[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + chop > 61.8 (ranging)
            long_signal = trix_cross_up[i] and volume_spike[i] and chop_high[i]
            
            # Short: TRIX crosses below zero + volume spike + chop > 61.8 (ranging)
            short_signal = trix_cross_down[i] and volume_spike[i] and chop_high[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR chop < 38.2 (trending starts) OR volume normalizes
            if trix_cross_down[i] or chop_values[i] < 38.2 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR chop < 38.2 (trending starts) OR volume normalizes
            if trix_cross_up[i] or chop_values[i] < 38.2 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0