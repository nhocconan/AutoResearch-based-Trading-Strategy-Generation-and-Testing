#!/usr/bin/env python3
name = "1d_WeeklyKAMA_Trend_12hRSI_Filter"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # === Weekly data for trend context ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 12h data for RSI filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === Daily KAMA calculation ===
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(change.reshape(-1, 10), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Fast=2, Slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 12h RSI (14) ===
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # === Align indicators to daily timeframe ===
    kama_aligned = kama  # Already daily
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Weekly trend: price above/below weekly EMA(20)
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA AND weekly uptrend AND RSI not overbought
            if (close[i] > kama_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and
                rsi_12h_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA AND weekly downtrend AND RSI not oversold
            elif (close[i] < kama_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and
                  rsi_12h_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below KAMA or weekly trend breaks
            if close[i] < kama_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above KAMA or weekly trend breaks
            if close[i] > kama_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals