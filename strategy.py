#!/usr/bin/env python3
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
    
    # Daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA calculation (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = abs(close - close[10]) / sum(abs(close - close[1])) over 10 periods
    change = np.abs(np.subtract(close_1d[10:], close_1d[:-10]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 0
    # Proper ER calculation for each point
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        price_change = np.abs(close_1d[i] - close_1d[i-10])
        price_volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        er[i] = price_change / price_volatility if price_volatility > 0 else 0
    er[:10] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_20_1w = np.convolve(close_1w, np.ones(20)/20, mode='same')
    # Handle edges
    sma_20_1w[:10] = close_1w[:10].mean() if len(close_1w) >= 10 else close_1w[0]
    sma_20_1w[-10:] = close_1w[-10:].mean() if len(close_1w) >= 10 else close_1w[-1]
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(sma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above KAMA = uptrend, below = downtrend
        kama_up = close[i] > kama_aligned[i]
        kama_down = close[i] < kama_aligned[i]
        
        # RSI conditions: avoid extremes, look for momentum
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = close[i] > sma_20_1w_aligned[i]
        weekly_downtrend = close[i] < sma_20_1w_aligned[i]
        
        if position == 0:
            # Long: price above KAMA + RSI bullish + weekly uptrend
            if kama_up and rsi_bullish and weekly_uptrend:
                position = 1
                signals[i] = position_size
            # Short: price below KAMA + RSI bearish + weekly downtrend
            elif kama_down and rsi_bearish and weekly_downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA or RSI turns bearish
            if not kama_up or not rsi_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price above KAMA or RSI turns bullish
            if not kama_down or not rsi_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_KAMA_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0