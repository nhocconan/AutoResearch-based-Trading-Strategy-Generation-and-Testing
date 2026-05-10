#!/usr/bin/env python3
# 6h_Keltner_MeanReversion_1dATR_Filter
# Hypothesis: Mean reversion at Keltner channel extremes filtered by daily ATR volatility regime.
# Long when price touches lower Keltner band (EMA20 - 2*ATR) with daily ATR > median (high volatility).
# Short when price touches upper Keltner band (EMA20 + 2*ATR) with daily ATR > median.
# Exit when price crosses EMA20. Works in both bull/bear markets by capturing overextended moves during high volatility.

name = "6h_Keltner_MeanReversion_1dATR_Filter"
timeframe = "6h"
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
    
    # Calculate 6h EMA20 and ATR(20)
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands
    upper_keltner = ema20 + 2.0 * atr
    lower_keltner = ema20 - 2.0 * atr
    
    # Calculate daily ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(20)
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily ATR median for regime filter (using expanding window to avoid look-ahead)
    atr_median_1d = np.full(len(atr_1d), np.nan)
    for i in range(20, len(atr_1d)):
        atr_median_1d[i] = np.nanmedian(atr_1d[20:i+1])
    
    # Align daily ATR and median to 6h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or np.isnan(ema20[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(atr_median_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # High volatility regime: daily ATR above its median
        high_vol = atr_1d_aligned[i] > atr_median_1d_aligned[i]
        
        if position == 0:
            # Long: Price at lower Keltner band in high volatility
            if low[i] <= lower_keltner[i] and high_vol:
                signals[i] = 0.25
                position = 1
            # Short: Price at upper Keltner band in high volatility
            elif high[i] >= upper_keltner[i] and high_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses above EMA20 (mean reversion complete)
            if close[i] >= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses below EMA20 (mean reversion complete)
            if close[i] <= ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals