#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (15-period) momentum with volume spike and choppiness regime filter.
Long when TRIX crosses above zero with volume spike in trending market (CHOP < 38.2).
Short when TRIX crosses below zero with volume spike in trending market (CHOP < 38.2).
ATR trailing stop (2.0) and discrete sizing (0.25) to limit trades (~20-50/year).
Designed to work in both bull/bear via momentum confirmation + volatility regime filter.
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
    
    # TRIX calculation: EMA(EMA(EMA(close,15),15),15) then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (pd.Series(ema3).pct_change().values)
    
    # 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                            np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HHV - LLV)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hhvs = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llvs = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_atr_14 / (hhvs - llvs + 1e-10)) / np.log10(14))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need TRIX (45), CHOP aligned (14+13), volume MA (20), ATR (14)
    start_idx = max(45, 27, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        prev_trix = trix[i-1]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike in trending market (CHOP < 38.2)
            long_signal = (prev_trix <= 0 and trix[i] > 0) and vol_spike[i] and (chop_aligned[i] < 38.2)
            # Short: TRIX crosses below zero with volume spike in trending market (CHOP < 38.2)
            short_signal = (prev_trix >= 0 and trix[i] < 0) and vol_spike[i] and (chop_aligned[i] < 38.2)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR chop becomes too high OR ATR stoploss hit
            if (prev_trix >= 0 and trix[i] < 0) or (chop_aligned[i] > 61.8) or (curr_close < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR chop becomes too high OR ATR stoploss hit
            if (prev_trix <= 0 and trix[i] > 0) or (chop_aligned[i] > 61.8) or (curr_close > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0