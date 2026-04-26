#!/usr/bin/env python3
"""
4h_TRIX_9_VolumeSpike_ChopFilter_Regime
Hypothesis: 4h TRIX(9) zero-line cross with volume confirmation (>2.0x 20-period MA) and chop regime filter (CHOP > 61.8 = range, mean reversion). 
Long when TRIX crosses above zero in choppy market with volume spike. Short when TRIX crosses below zero in choppy market with volume spike.
TRIX is a momentum oscillator that filters out insignificant cycles and is effective in ranging markets. Volume spike confirms participation.
Chop filter ensures we only trade in ranging conditions where mean reversion works. Uses discrete position sizing (0.25) to minimize fee churn.
Designed for 20-50 trades/year on 4h timeframe. Works in both bull and bear markets by adapting to regime.
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
    
    # Calculate TRIX(9): triple EMA of close, then ROC
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=9, min_periods=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, min_periods=9, adjust=False).mean()
    ema3 = ema2.ewm(span=9, min_periods=9, adjust=False).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # Zero-line cross signals
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = 0
    trix_cross_up = (trix > 0) & (trix_prev <= 0)
    trix_cross_down = (trix < 0) & (trix_prev >= 0)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index regime filter: CHOP > 61.8 = ranging (mean reversion zone)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest(high,14) - lowest(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = highest_high14 - lowest_low14
    
    # Avoid division by zero
    mask = (range14 > 0) & (atr14 > 0)
    chop = np.full(n, 50.0)  # default neutral
    chop[mask] = 100 * np.log10(atr14[mask] / range14[mask]) / np.log10(14)
    chop_regime = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume, 14 for chop, 9 for TRIX)
    start_idx = max(20, 14, 9)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero in choppy market with volume spike
            if (trix_cross_up[i] and chop_regime[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero in choppy market with volume spike
            elif (trix_cross_down[i] and chop_regime[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR chop regime ends (trending market)
            if (trix_cross_down[i] or not chop_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR chop regime ends (trending market)
            if (trix_cross_up[i] or not chop_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_9_VolumeSpike_ChopFilter_Regime"
timeframe = "4h"
leverage = 1.0