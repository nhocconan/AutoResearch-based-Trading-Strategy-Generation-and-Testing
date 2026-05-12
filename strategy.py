#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_1dTrend
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 12h captures trend direction with lower whipsaw.
# Enter long when price > KAMA and RSI < 50 in bullish 1d trend (price > 1d EMA50).
# Enter short when price < KAMA and RSI > 50 in bearish 1d trend (price < 1d EMA50).
# Uses 1d trend filter to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Designed for low frequency (~20-40 trades/year) to survive both bull and bear markets.

name = "12h_KAMA_Trend_RSI_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h KAMA (adaptive trend) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_12h, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=1)  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan, dtype=float)
    kama[19] = close_12h[19]  # seed
    for i in range(20, len(close_12h)):
        kama[i] = kama[i-1] + sc[i-10] * (close_12h[i] - kama[i-1])
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === RSI(14) on 12h ===
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # KAMA position
        price_above_kama = close[i] > kama_12h_aligned[i]
        price_below_kama = close[i] < kama_12h_aligned[i]
        
        # RSI condition: avoid extremes, favor mean reversion within trend
        rsi_below_50 = rsi_12h_aligned[i] < 50
        rsi_above_50 = rsi_12h_aligned[i] > 50
        
        if position == 0:
            # LONG: price > KAMA, uptrend, RSI < 50 (not overbought)
            if price_above_kama and trend_up and rsi_below_50:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA, downtrend, RSI > 50 (not oversold)
            elif price_below_kama and trend_down and rsi_above_50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price < KAMA or trend reversal
            if price_below_kama or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > KAMA or trend reversal
            if price_above_kama or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals