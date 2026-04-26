#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_1dTrend_v1
Hypothesis: TRIX momentum with volume confirmation and 1d choppiness regime filter for 4h timeframe.
Long when TRIX crosses above zero with volume spike and chop regime > 61.8 (ranging).
Short when TRIX crosses below zero with volume spike and chop regime > 61.8.
Uses 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50.
Designed for low trade frequency (<50/year) with strong edge in ranging markets.
Works in both bull and bear markets by adapting to regime (choppy = mean reversion via TRIX zero crosses).
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
    
    # Calculate TRIX (15-period EMA of EMA of EMA of close, then ROC)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = pd.Series(ema3).pct_change(periods=1) * 100
    trix_values = trix.values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Load 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (n * log(n))) / log10(n)
    # Simplified: use TR and rolling sum
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr1 = np.maximum(tr1, 
                     np.absolute(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])))
    atr1 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean()
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum()
    high_low_diff14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(sum_atr14 / high_low_diff14) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(15+15+15+1, 20, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_values[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: TRIX crosses above zero + volume spike + chop > 61.8 (ranging) + price > 1d EMA50
        trix_cross_up = trix_values[i] > 0 and trix_values[i-1] <= 0
        in_choppy_regime = chop_aligned[i] > 61.8
        if trix_cross_up and volume_spike[i] and in_choppy_regime and close[i] > ema_50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TRIX crosses below zero + volume spike + chop > 61.8 (ranging) + price < 1d EMA50
        elif trix_values[i] < 0 and trix_values[i-1] >= 0 and volume_spike[i] and in_choppy_regime and close[i] < ema_50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: TRIX crosses zero in opposite direction or loss of volume/chop regime
        elif position == 1 and (trix_values[i] < 0 or not volume_spike[i] or chop_aligned[i] <= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (trix_values[i] > 0 or not volume_spike[i] or chop_aligned[i] <= 61.8):
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

name = "4h_TRIX_VolumeSpike_ChopRegime_1dTrend_v1"
timeframe = "4h"
leverage = 1.0