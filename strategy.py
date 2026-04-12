#!/usr/bin/env python3
"""
6h_1d_cci_ema_hybrid
Uses 1d EMA for trend direction and 6h CCI for momentum entries.
Long when 1d EMA200 uptrend + 6h CCI crosses above -100 (momentum shift).
Short when 1d EMA200 downtrend + 6h CCI crosses below +100.
Exit when CCI crosses back toward zero or EMA trend weakens.
Designed for low trade frequency (target: 10-25 trades/year) with clear trend-momentum alignment.
Works in bull (follow EMA uptrend) and bear (fade rallies in downtrend).
"""

name = "6h_1d_cci_ema_hybrid"
timeframe = "6h"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA200 on 1d for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # EMA50 on 1d for trend strength (optional filter)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # CCI(20) on 6h for momentum
    tp = (high + low + close) / 3.0  # typical price
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(cci[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA
        uptrend = close_1d[i // 24] > ema200_1d[i // 24] if i // 24 < len(close_1d) else False
        downtrend = close_1d[i // 24] < ema200_1d[i // 24] if i // 24 < len(close_1d) else False
        
        # Long: uptrend + CCI crosses above -100 (bullish momentum)
        if uptrend and cci[i] > -100 and cci[i-1] <= -100 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: downtrend + CCI crosses below +100 (bearish momentum)
        elif downtrend and cci[i] < 100 and cci[i-1] >= 100 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI crosses back toward zero or trend weakens
        elif position == 1 and (cci[i] < 0 or close_1d[i // 24] < ema200_1d[i // 24]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci[i] > 0 or close_1d[i // 24] > ema200_1d[i // 24]):
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