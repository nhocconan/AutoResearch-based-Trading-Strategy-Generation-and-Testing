#!/usr/bin/env python3
# 4h_Trix_Zero_Cross_Volume_Spike_Trend_Filter
# Hypothesis: TRIX zero-cross signals with volume spike confirmation and 1d trend filter.
# Works in bull/bear: Trend filter ensures we only trade with higher timeframe momentum.
# TRIX captures momentum changes; zero-cross provides objective entry/exit signals.
# Volume spike filters for institutional participation. Target: 20-40 trades/year.

name = "4h_Trix_Zero_Cross_Volume_Spike_Trend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate TRIX (12-period EMA of EMA of EMA of close, then ROC)
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    trix = np.concatenate([[np.nan], trix])  # align with original length
    
    # Calculate 1d EMA34 for trend filter (higher timeframe trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12)  # Ensure TRIX and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND uptrend (price > EMA34) AND volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND downtrend (price < EMA34) AND volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reversal (price < EMA34)
            if trix[i] < 0 and trix[i-1] >= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reversal (price > EMA34)
            if trix[i] > 0 and trix[i-1] <= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals