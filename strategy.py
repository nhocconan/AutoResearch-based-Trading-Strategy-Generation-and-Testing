#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_Chop_Regime
# Hypothesis: On 4h timeframe, use TRIX (triple-smoothed EMA) with volume spike and Choppiness index regime filter.
# Go long when TRIX turns positive, volume > 2x average, and market is trending (CHOP < 38.2).
# Go short when TRIX turns negative, volume > 2x average, and market is trending (CHOP < 38.2).
# Exit when TRIX crosses zero or volatility regime shifts to choppy (CHOP > 61.8).
# This avoids whipsaw in sideways markets and captures momentum in trending regimes.
# Targets 25-40 trades/year to minimize fee drag.

name = "4h_TRIX_Volume_Spike_Chop_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX calculation (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value has no previous
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness index (14-period)
    def calculate_chop(high, low, close, window=14):
        atr = []
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        for i in range(1, len(tr)):
            atr.append(np.mean(tr[max(0, i-window+1):i+1]))
        atr = np.array(atr)
        sum_atr = np.convolve(atr, np.ones(window)/window, mode='same')
        max_hh = np.maximum.accumulate(high)
        min_ll = np.minimum.accumulate(low)
        range_max_min = max_hh - min_ll
        chop = 100 * np.log10(sum_atr / range_max_min) / np.log10(window)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(trix[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        trix_val = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        vol_ratio_val = vol_ratio[i]
        chop_val = chop[i]
        
        if position == 0:
            # LONG: TRIX turns positive, volume spike, trending market (CHOP < 38.2)
            if trix_val > 0 and trix_prev <= 0 and vol_ratio_val > 2.0 and chop_val < 38.2:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX turns negative, volume spike, trending market (CHOP < 38.2)
            elif trix_val < 0 and trix_prev >= 0 and vol_ratio_val > 2.0 and chop_val < 38.2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses zero down OR market becomes choppy (CHOP > 61.8)
            if trix_val < 0 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses zero up OR market becomes choppy (CHOP > 61.8)
            if trix_val > 0 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals