#!/usr/bin/env python3
# 4h_Keltner_MeanReversion_1dATR_Filter
# Hypothesis: Mean reversion at Keltner Channel lower/upper bands with ATR-based volatility filter.
# Long when price touches lower Keltner band (EMA20 - 2*ATR) and 1d ATR ratio (current/20-period) > 1.5.
# Short when price touches upper Keltner band (EMA20 + 2*ATR) and 1d ATR ratio > 1.5.
# Exit when price returns to EMA20.
# Designed for 25-40 trades/year to avoid overtrading and work in both bull and bear markets.

name = "4h_Keltner_MeanReversion_1dATR_Filter"
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
    
    # Calculate 1d ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(20) on 1d
    atr_20_1d = np.full(len(tr_1d), np.nan)
    for i in range(20, len(tr_1d)):
        atr_20_1d[i] = np.mean(tr_1d[i-20:i])
    
    # Current ATR (last value) and 20-period average for ratio
    current_atr_1d = atr_20_1d[-1] if len(atr_20_1d) > 0 else np.nan
    atr_avg_20_1d = np.full(len(tr_1d), np.nan)
    for i in range(20, len(tr_1d)):
        atr_avg_20_1d[i] = np.mean(tr_1d[i-20:i])
    
    # ATR ratio: current ATR / 20-period average ATR
    atr_ratio_1d = np.full(len(tr_1d), np.nan)
    for i in range(20, len(tr_1d)):
        if atr_avg_20_1d[i] > 0:
            atr_ratio_1d[i] = atr_20_1d[i] / atr_avg_20_1d[i]
    
    # Align ATR ratio to 4h timeframe (needs 2-bar delay for ATR confirmation)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d, additional_delay_bars=2)
    
    # Keltner Channel on 4h: EMA(20) ± 2*ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range for 4h
    tr1_4h = np.abs(high - low)
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    
    # ATR(10) on 4h
    atr_10_4h = np.full(len(tr_4h), np.nan)
    for i in range(10, len(tr_4h)):
        atr_10_4h[i] = np.mean(tr_4h[i-10:i])
    
    # Keltner Bands
    keltner_lower = ema_20 - 2.0 * atr_10_4h
    keltner_upper = ema_20 + 2.0 * atr_10_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # EMA20 and ATR10 warmup
    
    for i in range(start_idx, n):
        if np.isnan(keltner_lower[i]) or np.isnan(keltner_upper[i]) or np.isnan(atr_ratio_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price at lower Keltner band with high volatility regime (ATR ratio > 1.5)
            if close[i] <= keltner_lower[i] and atr_ratio_1d_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Price at upper Keltner band with high volatility regime (ATR ratio > 1.5)
            elif close[i] >= keltner_upper[i] and atr_ratio_1d_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to EMA20
            if close[i] >= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to EMA20
            if close[i] <= ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals