#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Momentum
# Hypothesis: 12h chart strategy using KAMA for adaptive trend detection, RSI for momentum, and volume confirmation.
# KAMA adapts to market noise, reducing false signals in choppy markets. RSI filters for momentum strength.
# Volume confirmation ensures breakouts have institutional participation. Works in bull/bear via trend filter.
# Target: 15-35 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "12h"
name = "12h_KAMA_Trend_RSI_Momentum"
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
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA calculation (adaptive moving average)
    def kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 0 else np.abs(np.diff(close))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate indicators
    kama_vals = kama(close)
    rsi_vals = rsi(close)
    
    # Volume spike detection: 2x average volume (24-period = 2 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Ensure we have EMA50, KAMA, RSI, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI > 50 (bullish momentum), volume surge
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] > 50 and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI < 50 (bearish momentum), volume surge
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] < 50 and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA (trend change)
            if close[i] < kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA (trend change)
            if close[i] > kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals