#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_Trend
# Hypothesis: 12h timeframe with KAMA direction filter and RSI trend filter.
# KAMA adapts to market conditions, reducing whipsaws in choppy markets.
# RSI confirms momentum direction (RSI > 50 for long, RSI < 50 for short).
# Designed for 12h timeframe to target 15-30 trades/year per symbol.
# Works in bull/bear by requiring both KAMA and RSI to agree on trend.

name = "12h_KAMA_Direction_RSI_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for KAMA and RSI
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: price > KAMA and RSI > 50 for long, price < KAMA and RSI < 50 for short
        kama_trend = close[i] > kama_aligned[i]
        rsi_bullish = rsi_aligned[i] > 50
        kama_downtrend = close[i] < kama_aligned[i]
        rsi_bearish = rsi_aligned[i] < 50
        
        if position == 0:
            # Long: price above KAMA and RSI bullish
            if kama_trend and rsi_bullish:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI bearish
            elif kama_downtrend and rsi_bearish:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price below KAMA or RSI turns bearish
                if not kama_trend or not rsi_bullish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price above KAMA or RSI turns bullish
                if not kama_downtrend or not rsi_bearish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals