#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Pullback_1wTrend_Filter
Hypothesis: Use KAMA trend direction on 1d with RSI(14) pullback entries (long when RSI<30 in uptrend, short when RSI>70 in downtrend) and 1w EMA50 trend filter to avoid counter-trend trades. Designed to capture mean reversion within stronger trends, working in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets. Target 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate KAMA trend on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Efficiency Ratio for KAMA
    change = abs(df_1d['close'].diff(10)).values
    volatility = df_1d['close'].diff().abs().rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(df_1d['close'].values, np.nan, dtype=float)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 1d timeframe (wait for previous day's close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50, additional_delay_bars=1)
    
    # RSI(14) on 1d
    delta = df_1d['close'].diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA, EMA, RSI
    start_idx = max(30, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        ema_50_val = ema_50_aligned[i]
        rsi_val = rsi_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI < 30 (oversold), and above 1w EMA50 (bullish higher tf)
            if close_val > kama_val and rsi_val < 30 and close_val > ema_50_val:
                signals[i] = size
                position = 1
            # Short: price below KAMA (downtrend), RSI > 70 (overbought), and below 1w EMA50 (bearish higher tf)
            elif close_val < kama_val and rsi_val > 70 and close_val < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA (trend change) or RSI > 70 (overbought)
            if close_val < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA (trend change) or RSI < 30 (oversold)
            if close_val > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_Pullback_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0