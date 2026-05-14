#!/usr/bin/env python3
"""
12h_1d_ema_crossover_volatility_filter
Hypothesis: 12-hour strategy using EMA crossovers on daily timeframe for trend direction, with volatility-based position sizing.
Uses daily EMA(50) and EMA(200) crossover for trend, and ATR-based volatility filter to avoid choppy markets.
Designed to work in both bull and bear markets by only taking trades aligned with the higher timeframe trend.
Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA50 and EMA200 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily ATR for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12-period ATR for position sizing (using 12h data)
    tr_12h1 = np.abs(high - low)
    tr_12h2 = np.abs(np.roll(high, 1) - close)
    tr_12h3 = np.abs(np.roll(low, 1) - close)
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    atr_12h = pd.Series(tr_12h).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median (avoid choppy markets)
        atr_median = np.nanmedian(atr_12h[max(0, i-50):i+1])
        if atr_12h[i] < atr_median * 0.8:  # Avoid low volatility periods
            # Hold current position or stay flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend determination: EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Entry conditions
        if uptrend and position != 1:
            # Long entry: pullback to EMA50 area with momentum
            if close[i] > ema50_1d_aligned[i] * 0.995 and close[i] < ema50_1d_aligned[i] * 1.005:
                position = 1
                signals[i] = 0.25  # Fixed size to reduce churn
        elif downtrend and position != -1:
            # Short entry: rally to EMA50 area with momentum
            if close[i] < ema50_1d_aligned[i] * 1.005 and close[i] > ema50_1d_aligned[i] * 0.995:
                position = -1
                signals[i] = -0.25  # Fixed size to reduce churn
        # Exit conditions: trend reversal
        elif position == 1 and downtrend:
            position = 0
            signals[i] = 0.0
        elif position == -1 and uptrend:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_ema_crossover_volatility_filter"
timeframe = "12h"
leverage = 1.0