#!/usr/bin/env python3
"""
6h Adaptive Keltner Channel with 12h Trend Filter
Hypothesis: In trending markets (12h EMA25 > EMA50), price tends to respect the 2xATR Keltner Channel.
Buy near lower band (1.01x) in uptrend, sell near upper band (0.99x) in downtrend.
The adaptive channel width (ATR-based) handles volatility regimes, reducing whipsaws in chop.
Designed for 15-35 trades/year to minimize fee drag while capturing trend persistence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adaptive_keltner_12h_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA25 and EMA50 for trend filter
    ema_25_12h = pd.Series(close_12h).ewm(span=25, min_periods=25, adjust=False).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(20) for Keltner Channel width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel: 2 * ATR around EMA22
    ema22 = pd.Series(close).ewm(span=22, min_periods=22, adjust=False).mean().values
    kc_upper = ema22 + 2.0 * atr
    kc_lower = ema22 - 2.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_25_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: 1=up (EMA25 > EMA50), -1=down (EMA25 < EMA50)
        if ema_25_12h_aligned[i] > ema_50_12h_aligned[i]:
            trend = 1
        elif ema_25_12h_aligned[i] < ema_50_12h_aligned[i]:
            trend = -1
        else:
            trend = 0
        
        if position == 1:  # Long position
            # Exit: trend turns down OR price reaches upper band
            if trend == -1 or close[i] >= kc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns up OR price reaches lower band
            if trend == 1 or close[i] <= kc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price near lower band in uptrend
            if trend == 1 and close[i] <= kc_lower[i] * 1.01:
                position = 1
                signals[i] = 0.25
            # Short: price near upper band in downtrend
            elif trend == -1 and close[i] >= kc_upper[i] * 0.99:
                position = -1
                signals[i] = -0.25
    
    return signals