#!/usr/bin/env python3
# 4h_KAMA_RSI_Trend
# Hypothesis: KAMA identifies the trend direction while RSI measures momentum strength
# Works in bull markets via strong up-trends and in bear markets via strong down-trends
# RSI filters out weak momentum to reduce false signals
# Target: 15-30 trades per year (~60-120 over 4 years) with position size 0.25

name = "4h_KAMA_RSI_Trend"
timeframe = "4h"
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
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Use rolling window for volatility sum
    volatility_sum = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=er_length, min_periods=1).sum().values
    price_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.where(volatility_sum > 0, price_change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1d data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, er_length)  # Need enough data for RSI and KAMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # RSI momentum filter
        strong_momentum = (rsi[i] > 60) or (rsi[i] < 40)
        
        if position == 0:
            # Long: price above KAMA and EMA with strong upward momentum
            if price_above_kama and price_above_ema and rsi[i] > 60:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and EMA with strong downward momentum
            elif price_below_kama and price_below_ema and rsi[i] < 40:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or trend reversal
            if close[i] < kama[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or trend reversal
            if close[i] > kama[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals