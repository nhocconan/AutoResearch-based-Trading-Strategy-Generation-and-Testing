#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: 4h TRIX (triple EMA) momentum with volume spike (>2.0x 20-bar avg) and choppiness regime filter (CHOP(14) > 61.8 = ranging) captures mean-reversion bounces in sideways markets, which dominates BTC/ETH in 2025 bear/range conditions. Long when TRIX crosses above signal line with volume spike in choppy market; short when TRIX crosses below signal line with volume spike in choppy market. Uses discrete position sizing (0.25) to limit fee drag. Designed for ~30-50 trades/year to avoid overtrading.
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
    
    # TRIX calculation: triple EMA of close, then ROC
    def ema(series, span):
        return pd.Series(series).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    # Avoid division by zero
    ema3_prev = np.roll(ema3, 1)
    ema3_prev[0] = ema3[0] if not np.isnan(ema3[0]) else 0.0
    trix = 100 * (ema3 - ema3_prev) / ema3_prev
    # Signal line: EMA of TRIX
    trix_signal = ema(trix, 9)
    # TRIX histogram: TRIX - signal
    trix_hist = trix - trix_signal
    
    # Choppiness Index: CHOP(14) = 100 * log10(sum(ATR(1)) / (n * (HH(14) - LL(14)))) / log10(n)
    # Simplified: high-low range over 14 periods
    atr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    chop = np.zeros_like(close)
    mask = (range_14 > 0) & (~np.isnan(sum_atr1))
    chop[mask] = 100 * np.log10(sum_atr1[mask] / (14 * range_14[mask])) / np.log10(14)
    chop_mask = chop > 61.8  # choppy/ranging market
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for TRIX, CHOP, volume MA
    start_idx = max(12+12+12+9, 14, 20)  # TRIX: 33, CHOP: 14, Volume: 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume spike in choppy market
            long_setup = (trix[i] > trix_signal[i]) and (trix[i-1] <= trix_signal[i-1]) and volume_spike[i] and chop_mask[i]
            # Short: TRIX crosses below signal line with volume spike in choppy market
            short_setup = (trix[i] < trix_signal[i]) and (trix[i-1] >= trix_signal[i-1]) and volume_spike[i] and chop_mask[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TRIX crosses below signal line OR volume spike disappears
            if (trix[i] < trix_signal[i]) and (trix[i-1] >= trix_signal[i-1]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX crosses above signal line OR volume spike disappears
            if (trix[i] > trix_signal[i]) and (trix[i-1] <= trix_signal[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0