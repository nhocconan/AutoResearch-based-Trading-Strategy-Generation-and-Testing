#!/usr/bin/env python3
"""
1d Bollinger Band Breakout with 1w Trend Filter
Long when price breaks above upper band and 1w EMA21 > EMA50 (bullish)
Short when price breaks below lower band and 1w EMA21 < EMA50 (bearish)
Exit when price returns to middle band
Designed to capture breakouts in trending markets with volatility filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_bollinger_breakout_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Bollinger Bands (20, 2) ===
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    basis = basis.values
    upper = upper.values
    lower = lower.values
    
    # === 1w EMA Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_21 = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False).mean().values
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or \
           np.isnan(ema_21_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle band
            if close[i] <= basis[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price returns to middle band
            if close[i] >= basis[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Bullish trend: EMA21 > EMA50
            if ema_21_aligned[i] > ema_50_aligned[i]:
                # Bullish trend - look for long breakout
                if close[i] > upper[i]:
                    position = 1
                    signals[i] = 0.30
            # Bearish trend: EMA21 < EMA50
            elif ema_21_aligned[i] < ema_50_aligned[i]:
                # Bearish trend - look for short breakdown
                if close[i] < lower[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals