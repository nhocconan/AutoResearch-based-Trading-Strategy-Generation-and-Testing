#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA200 trend filter and ATR-based volatility filter.
Long when Bull Power > 0 AND price > 1d EMA200 AND ATR(6h) > 0.5 * ATR(1d) (sufficient volatility).
Short when Bear Power < 0 AND price < 1d EMA200 AND ATR(6h) > 0.5 * ATR(1d).
Exit when Bull/Bear Power crosses zero OR ATR(6h) < 0.3 * ATR(1d) (low volatility).
Elder Ray measures buying/selling pressure relative to EMA13. EMA200 filter ensures we trade with the higher timeframe trend.
ATR ratio filter avoids choppy markets. Works in bull markets (captures strong uptrends) and bear markets (captures strong downtrends).
Target: 50-150 total trades over 4 years (12-37/year). Discrete sizing 0.25 minimizes fee churn.
"""

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
    
    # Get 1d data for EMA200 and ATR
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA200 on 1d
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) on 1d
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d EMA200 and ATR to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate ATR(14) on 6h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_6h[0] = tr1[0]  # first period
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA13 on 6h for Elder Ray
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        ema200 = ema200_1d_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_6h_val = atr_6h[i]
        bull = bull_power[i]
        bear = bear_power[i]
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price > 1d EMA200 AND sufficient volatility (ATR ratio > 0.5)
            if bull > 0 and price > ema200 and atr_6h_val > 0.5 * atr_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price < 1d EMA200 AND sufficient volatility (ATR ratio > 0.5)
            elif bear < 0 and price < ema200 and atr_6h_val > 0.5 * atr_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR low volatility (ATR ratio < 0.3)
            if bull <= 0 or atr_6h_val < 0.3 * atr_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 OR low volatility (ATR ratio < 0.3)
            if bear >= 0 or atr_6h_val < 0.3 * atr_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA200_ATRFilter"
timeframe = "6h"
leverage = 1.0