#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Overbought_Oversold
# Hypothesis: Daily KAMA trend filter with RSI extremes for mean reversion in ranging markets, avoiding counter-trend trades.
# KAMA adapts to market noise, reducing false signals during high volatility. RSI < 30 for long, RSI > 70 for short.
# Trend filter ensures trades align with higher timeframe (weekly) direction. Designed for 1d to achieve 7-25 trades/year.

name = "1d_KAMA_Trend_RSI_Overbought_Oversold"
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
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily KAMA (trend component)
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(1, len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # Daily RSI
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Initial average
        if len(close) >= period:
            avg_gain[period-1] = np.mean(gain[1:period])
            avg_loss[period-1] = np.mean(loss[1:period])
            
            # Wilder smoothing
            for i in range(period, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(close)
        rsi = np.zeros_like(close)
        for i in range(period, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Align weekly indicators to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) and RSI oversold
            if close[i] > kama[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) and RSI overbought
            elif close[i] < kama[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or price below KAMA
            if rsi[i] > 70 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or price above KAMA
            if rsi[i] < 30 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals