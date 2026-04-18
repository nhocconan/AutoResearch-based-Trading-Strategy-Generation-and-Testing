#!/usr/bin/env python3
"""
1d_1W_KAMA_RSI_Trend_Filter
Hypothesis: On daily chart, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
filtered by RSI momentum and weekly trend filter to avoid false signals in sideways markets.
Long when KAMA turns up, RSI > 50, and weekly close > weekly EMA34.
Short when KAMA turns down, RSI < 50, and weekly close < weekly EMA34.
Position size 0.25 to manage drawdown. Designed for low trade frequency (<20/year) to minimize fee drag.
Works in bull/bear via dual timeframe alignment and momentum filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily KAMA calculation
    # Efficiency ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Proper ER calculation over 10-period window
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
        er[i] = price_change / price_volatility if price_volatility != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first 14 values
    rsi = np.concatenate([np.full(14, 50), rsi[14:]])
    
    # Align KAMA and RSI to daily (already aligned as same timeframe)
    # But ensure we have proper alignment for signals
    kama_aligned = kama  # same timeframe
    rsi_aligned = rsi    # same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 10)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA over 2 periods
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        if position == 0:
            # Long: KAMA turning up, RSI > 50, weekly close > weekly EMA34
            if kama_up and rsi_aligned[i] > 50 and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI < 50, weekly close < weekly EMA34
            elif kama_down and rsi_aligned[i] < 50 and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI < 40 or weekly trend fails
            if (not kama_up) or rsi_aligned[i] < 40 or close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI > 60 or weekly trend fails
            if (not kama_down) or rsi_aligned[i] > 60 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_KAMA_RSI_Trend_Filter"
timeframe = "1d"
leverage = 1.0