#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_ChopRegime
# Hypothesis: TRIX (12-period triple exponential smoothing) captures momentum with less whipsaw than MACD.
# Long when TRIX crosses above zero with volume spike in choppy market (CHOP > 61.8), short when TRIX crosses below zero with volume spike in choppy market.
# Chop regime filter prevents trending whipsaw; volume spike confirms momentum breakout.
# Target: 20-40 trades/year on 4h timeframe for low fee drag and robustness in bull/bear markets.

name = "4h_TRIX_VolumeSpike_ChopRegime"
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
    
    # === TRIX (12,12,12) ===
    # Single EMA
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Double EMA
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Triple EMA
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = percent change of triple EMA
    trix = np.zeros_like(close)
    trix[12:] = (ema3[12:] - ema3[11:-1]) / ema3[11:-1] * 100
    
    # === Chopiness Index (14-period) ===
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    denom = atr14 * 14
    chop[13:] = np.where(denom[13:] > 0, 
                         100 * np.log10(max_high[13:] - min_low[13:]) / np.log10(denom[13:]), 
                         50)
    
    # === Volume Spike (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: choppy market (CHOP > 61.8 = ranging)
        choppy = chop[i] > 61.8
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma_20[i] * 1.5  # 50% above average
        
        # TRIX signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        if position == 0:
            # LONG: TRIX crosses above zero in choppy market with volume spike
            if trix_cross_up and choppy and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero in choppy market with volume spike
            elif trix_cross_down and choppy and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or chop breaks down (trending)
            if trix_cross_down or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or chop breaks down (trending)
            if trix_cross_up or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals