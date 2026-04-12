#!/usr/bin/env python3
"""
1d_1w_KAMA_Reversal_Strategy
Hypothesis: Kaufman Adaptive Moving Average (KAMA) on daily timeframe filters trend direction.
Weekly timeframe provides higher timeframe trend filter (EMA50).
Entries occur when price crosses KAMA with volume confirmation and price is near weekly EMA50 (mean reversion).
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Low trade frequency (~15-25/year) minimizes fee fade while capturing mean reversion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Reversal_Strategy"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY KAMA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # KAMA parameters: ER period=10, Fast EMA=2, Slow EMA=30
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already aligned as it's daily data)
    kama_aligned = kama
    
    # === WEEKLY EMA50 FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === VOLUME CONFIRMATION (DAILY) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price crosses above KAMA (bullish reversal) in weekly uptrend
        long_signal = (close[i] > kama_aligned[i]) and (close[i-1] <= kama_aligned[i-1]) and \
                      (close[i] > ema50_1w_aligned[i]) and (vol_ratio[i] > 1.3)
        
        # Short: price crosses below KAMA (bearish reversal) in weekly downtrend
        short_signal = (close[i] < kama_aligned[i]) and (close[i-1] >= kama_aligned[i-1]) and \
                       (close[i] < ema50_1w_aligned[i]) and (vol_ratio[i] > 1.3)
        
        # Exit when price returns to KAMA (mean reversion)
        exit_long = close[i] < kama_aligned[i] and position == 1
        exit_short = close[i] > kama_aligned[i] and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals