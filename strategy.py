#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and 1d volatility filter.
# Long when price > 1h EMA21 AND 4h EMA50 rising AND 1d ATR ratio > 1.2 (high volatility regime).
# Short when price < 1h EMA21 AND 4h EMA50 falling AND 1d ATR ratio > 1.2.
# Exit when price crosses back over 1h EMA21.
# This strategy captures momentum bursts during high volatility regimes, aligned with 4h trend.
# Volatility filter avoids choppy markets. EMA21 provides responsive entry/exit.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_EMA21_4hEMA50_1dATR_Volatility"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h EMA21 for entry/exit
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h EMA50 direction
    ema50_rising = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_4h_aligned[1:] > ema50_4h_aligned[:-1]
    ema50_falling[1:] = ema50_4h_aligned[1:] < ema50_4h_aligned[:-1]
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d ATR ratio: current ATR / 50-period average ATR
    atr_ma50 = pd.Series(atr14).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr14 / atr_ma50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 50)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema21[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > EMA21, 4h EMA50 rising, high volatility (ATR ratio > 1.2)
            long_cond = (close[i] > ema21[i]) and ema50_rising[i] and (atr_ratio_aligned[i] > 1.2)
            # Short conditions: price < EMA21, 4h EMA50 falling, high volatility (ATR ratio > 1.2)
            short_cond = (close[i] < ema21[i]) and ema50_falling[i] and (atr_ratio_aligned[i] > 1.2)
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below EMA21
            if close[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above EMA21
            if close[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals