#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Pullback_12hTrend_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for 4h trend direction, RSI(14) for pullback entries, and 12h EMA50 as higher timeframe filter. Long when KAMA slopes up and RSI < 30 with 12h EMA50 uptrend; short when KAMA slopes down and RSI > 70 with 12h EMA50 downtrend. Exit on opposite RSI extreme. Designed to capture trend continuations after pullbacks in both bull and bear markets. Target 20-30 trades/year to minimize fee drag.
"""

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
    
    # Calculate KAMA for trend direction (4h)
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 1 else np.sum(np.abs(np.diff(close)))
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close, length=10, fast=2, slow=30)
    kama_slope = np.diff(kama_val, prepend=0)
    
    # RSI for pullback entries
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_val = rsi(close, length=14)
    
    # 12h EMA50 for higher timeframe trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA, RSI, and EMA
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_slope_val = kama_slope[i]
        rsi_val_i = rsi_val[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: KAMA up (uptrend) + RSI < 30 (pullback) + 12h EMA50 uptrend
            if kama_slope_val > 0 and rsi_val_i < 30 and close[i] > ema_50_12h_val:
                signals[i] = size
                position = 1
            # Short: KAMA down (downtrend) + RSI > 70 (pullback) + 12h EMA50 downtrend
            elif kama_slope_val < 0 and rsi_val_i > 70 and close[i] < ema_50_12h_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 (overbought)
            if rsi_val_i > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 30 (oversold)
            if rsi_val_i < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_RSI_Pullback_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0