#!/usr/bin/env python3
name = "1d_KAMA_1wTrend_StockRSI"
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
    
    # KAMA on 1d (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    direction = np.abs(np.diff(close_1d, k=9, prepend=close_1d[:9]))
    er = np.where(change != 0, direction / change, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Stock RSI on 1d (momentum/mean reversion)
    rsi_period = 14
    stoch_period = 14
    k_smooth = 3
    d_smooth = 3
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_min = pd.Series(rsi).rolling(stoch_period, min_periods=stoch_period).min().values
    rsi_max = pd.Series(rsi).rolling(stoch_period, min_periods=stoch_period).max().values
    stoch_rsi = np.where((rsi_max - rsi_min) != 0, (rsi - rsi_min) / (rsi_max - rsi_min) * 100, 50)
    k = pd.Series(stoch_rsi).rolling(k_smooth, min_periods=1).mean().values
    d = pd.Series(k).rolling(d_smooth, min_periods=1).mean().values
    stock_rsi = d
    stock_rsi_1d = stock_rsi
    stock_rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, stock_rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(stock_rsi_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + Stock RSI oversold (<30)
            if close[i] > kama_1d_aligned[i] and stock_rsi_1d_aligned[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + Stock RSI overbought (>70)
            elif close[i] < kama_1d_aligned[i] and stock_rsi_1d_aligned[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below KAMA or Stock RSI overbought
            if close[i] < kama_1d_aligned[i] or stock_rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above KAMA or Stock RSI oversold
            if close[i] > kama_1d_aligned[i] or stock_rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals