#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_ChopFilter_v2
# Hypothesis: Uses TRIX (15-period) for momentum, volume confirmation, and Choppiness Index (14) for regime filtering.
# Long when TRIX > 0, volume > 2x 20-period average, and CHOP > 61.8 (ranging market).
# Short when TRIX < 0, volume > 2x 20-period average, and CHOP > 61.8.
# Exits when TRIX crosses zero or volume drops below average.
# Designed to work in both bull and bear markets by focusing on momentum bursts in ranging conditions.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_TRIX_VolumeSpike_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period) on close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100
    trix_values = trix.fillna(0).values  # Fill NaN with 0 for stability
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Choppiness Index (14) - requires high, low, close
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * ATR(1))) where n=14
    # Simplified: CHOP = 100 * log10(sum(tr_range_14) / (log10(14) * true_range))
    # We'll use common approximation: CHOP = 100 * log10(sum(ATR_14) / (log10(14) * ATR_current))
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = 14
    atr_values = pd.Series(true_range).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_sum = pd.Series(true_range).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_values = 100 * np.log10(atr_sum / (np.log10(atr_period) * atr_values))
    chop_values = np.where((atr_values == 0) | np.isinf(chop_values) | np.isnan(chop_values), 50, chop_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20)  # Warmup for TRIX (3*15=45) and volume MA (20)
    
    for i in range(start_idx, n):
        if np.isnan(trix_values[i]) or np.isnan(volume_confirm[i]) or np.isnan(chop_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop_values[i] > 61.8
        
        if position == 0:
            # Long entry: TRIX positive, volume spike, ranging market
            if trix_values[i] > 0 and volume_confirm[i] and in_range:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX negative, volume spike, ranging market
            elif trix_values[i] < 0 and volume_confirm[i] and in_range:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or volume drops
            if trix_values[i] <= 0 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or volume drops
            if trix_values[i] >= 0 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals