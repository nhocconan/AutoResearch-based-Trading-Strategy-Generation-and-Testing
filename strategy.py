#!/usr/bin/env python3
"""
6h_Keltner_Reversal_1dTrend_Volume
Hypothesis: Mean reversion on 6h using Keltner Channel (ATR-based) with 1d EMA trend filter and volume confirmation.
Works in bull/bear markets because: 1) In trends, price respects the Keltner mid-EMA as dynamic support/resistance, 2) 
In ranges, price reverts from upper/lower bands to the mean, 3) Volume spike confirms reversal strength, reducing false signals.
Target: 20-30 trades/year (80-120 total over 4 years).
"""
name = "6h_Keltner_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1d DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 6h KELTNER CHANNEL (20, 2.0) ===
    # EMA20 of close
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(20)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Keltner bands
    upper_keltner = ema20 + 2.0 * atr
    lower_keltner = ema20 - 2.0 * atr
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 34 for daily EMA, 20 for Keltner
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches lower Keltner band, price above daily EMA34, volume spike
            if (close[i] <= lower_keltner[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper Keltner band, price below daily EMA34, volume spike
            elif (close[i] >= upper_keltner[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses above EMA20 (mean) or below daily EMA34
            if (close[i] > ema20[i]) or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below EMA20 (mean) or above daily EMA34
            if (close[i] < ema20[i]) or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals