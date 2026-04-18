#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Hypothesis: TRIX (triple exponential moving average crossover) detects momentum shifts in 4h timeframe. 
Combined with volume spike (>2x 20-period average) and Choppiness Index regime filter (CHOP > 61.8 = ranging), 
the strategy enters long when TRIX turns positive in ranging markets and short when TRIX turns negative.
Designed to capture mean-reversion bounces in ranging conditions while avoiding strong trends where TRIX whipsaws.
Target: 20-40 trades/year with controlled risk via position sizing (0.25).
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
    
    # Calculate TRIX (15-period triple EMA of 1-period ROC)
    # ROC = (close - close.shift(1)) / close.shift(1)
    roc = np.zeros(n)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1]
    
    # Triple EMA of ROC
    def ema(series, period):
        result = np.full_like(series, np.nan)
        if len(series) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(series[:period])
        for i in range(period, len(series)):
            result[i] = series[i] * multiplier + result[i-1] * (1 - multiplier)
        return result
    
    ema1 = ema(roc, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix = ema3 * 100  # scale for readability
    
    # Calculate Choppiness Index (14-period)
    def choppiness_index(high, low, close, period=14):
        chop = np.full_like(high, np.nan)
        if len(high) < period:
            return chop
        atr = np.zeros_like(high)
        atr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = (atr[i-1] * (period-1) + tr) / period
        
        for i in range(period-1, len(high)):
            hh = np.max(high[i-period+1:i+1])
            ll = np.min(low[i-period+1:i+1])
            if hh - ll == 0:
                chop[i] = 50
            else:
                chop[i] = 100 * np.log10(atr[i] * period / (hh - ll)) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20, 14)  # TRIX needs ~36 bars for stability
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: TRIX turns positive in ranging market with volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and chop[i] > 61.8 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX turns negative in ranging market with volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and chop[i] > 61.8 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX turns negative or market becomes trending (CHOP < 38.2)
            if trix[i] < 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX turns positive or market becomes trending (CHOP < 38.2)
            if trix[i] > 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0