#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_BTC_ETH
Hypothesis: 4h strategy using TRIX (15,9) for momentum, volume spike (>2x 20-bar avg) for confirmation, and Choppiness Index (CHOP > 61.8) for ranging regime. Enters long when TRIX crosses above zero in choppy market with volume confirmation, short when TRIX crosses below zero. Uses 0.30 position size with discrete levels to minimize fee churn. Designed to work in both bull and bear markets by focusing on mean-reversion in ranging conditions (chop > 61.8) where TRIX zero-cross reversals are reliable. Targets 50-120 trades over 4 years (12-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- TRIX (15,9) --- #
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) then ROC of 9 periods
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = pd.Series(ema3).pct_change(periods=9).values * 100  # ROC 9
    
    # --- Volume Spike (>2x 20-bar average) --- #
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # --- Choppiness Index (CHOP, 14) --- #
    # CHOP = 100 * log10(sum(ATR(1)) / (HHV(high,14) - LLV(low,14))) / log10(14)
    atr1 = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    atr1[0] = high[0] - low[0]  # first bar
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = hh - ll
    # Avoid division by zero
    chop = np.where(denominator > 0, 100 * np.log10(sum_atr1 / denominator) / np.log10(14), 100)
    chop_regime = chop > 61.8  # ranging market
    
    # --- Signals --- #
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.30   # Position size
    
    # Warmup: need TRIX (15+15+15+9=54), vol avg (20), CHOP (14)
    start_idx = max(54, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_raw[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        trix_now = trix_raw[i]
        trix_prev = trix_raw[i-1]
        vol_spike = volume_spike[i]
        in_chop = chop_regime[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero in choppy market with volume spike
            if trix_prev <= 0 and trix_now > 0 and in_chop and vol_spike:
                signals[i] = size
                position = 1
            # Enter short: TRIX crosses below zero in choppy market with volume spike
            elif trix_prev >= 0 and trix_now < 0 and in_chop and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero (momentum loss)
            if trix_prev >= 0 and trix_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above zero (momentum loss)
            if trix_prev <= 0 and trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_BTC_ETH"
timeframe = "4h"
leverage = 1.0