#!/usr/bin/env python3
# 6h_TRIX_Volume_Spike_Chop_Regime
# Hypothesis: On 6h timeframe, use TRIX momentum with volume spike confirmation and chop regime filter.
# Long when TRIX crosses above zero, volume > 2x average, and chop > 61.8 (range) for mean reversion or chop < 38.2 (trend) for trend follow.
# Short when TRIX crosses below zero with same conditions.
# Exit when TRIX crosses back across zero.
# Works in bull markets via trend following and in bear via mean reversion in ranges.

name = "6h_TRIX_Volume_Spike_Chop_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # TRIX: triple EMA of ROC
    period = 12
    # ROC
    roc = np.diff(np.log(close), prepend=np.log(close[0])) * 100
    # EMA1
    ema1 = pd.Series(roc).ewm(span=period, adjust=False, min_periods=period).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
    # EMA3 (TRIX)
    ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Chop regime: chop = log(sum(tr, n)) / log(n) * 100
    # TR = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = np.log10(atr_sum / 14) / np.log10(14) * 100
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema3[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        trix_val = ema3[i]
        chop_val = chop[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: TRIX crosses above zero with volume > 2x average
            if trix_val > 0 and ema3[i-1] <= 0 and volume[i] > 2.0 * vol_ma_val:
                # In choppy market (chop > 61.8): mean reversion - wait for pullback
                # In trending market (chop < 38.2): trend follow - enter immediately
                if chop_val > 61.8:
                    # Wait for pullback to EMA(9) on 6h
                    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
                    if not np.isnan(ema9[i]) and close[i] < ema9[i]:
                        signals[i] = 0.25
                        position = 1
                    else:
                        signals[i] = 0.0
                else:
                    # Trending market - enter on TRIX cross
                    signals[i] = 0.25
                    position = 1
            # SHORT: TRIX crosses below zero with volume > 2x average
            elif trix_val < 0 and ema3[i-1] >= 0 and volume[i] > 2.0 * vol_ma_val:
                if chop_val > 61.8:
                    # Wait for pullback to EMA(9) on 6h
                    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
                    if not np.isnan(ema9[i]) and close[i] > ema9[i]:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    # Trending market - enter on TRIX cross
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix_val < 0 and ema3[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix_val > 0 and ema3[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals