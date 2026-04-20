#!/usr/bin/env python3
# 4h_1d_KAMA_RSI_TrendFilter
# Hypothesis: KAMA trend direction + RSI momentum on 4h with 1d ATR volatility filter.
# Uses KAMA (adaptive moving average) to capture trend with less whipsaw, RSI for momentum confirmation,
# and 1d ATR to avoid low-volatility chop. Designed for 20-35 trades/year per symbol.
# Works in bull via trend following, in bear via momentum reversals at extremes.

name = "4h_1d_KAMA_RSI_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_end = 2
    slow_end = 30
    
    # Calculate ER (Efficiency Ratio) and SSC (Smoothing Constant)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder
    
    # Proper volatility calculation (sum of absolute changes over period)
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    change_abs = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.where(volatility != 0, change_abs / volatility, 0)
    
    # SSC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2 / (fast_end + 1)
    slowest = 2 / (slow_end + 1)
    ss = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(ss[i]):
            kama[i] = kama[i-1] + ss[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # Wait for 1d close
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)   # Wait for 1d close
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)  # Wait for 1d close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > 50th percentile of recent ATR
        # Simplified: use current ATR > 0.5 * 20-period average ATR
        atr_ma = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr_1d_aligned[i] > 0.5 * atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        if position == 0:
            # Long: price above KAMA (uptrend) AND RSI > 50 (bullish momentum)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) AND RSI < 50 (bearish momentum)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA OR RSI < 40 (losing momentum)
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA OR RSI > 60 (losing momentum)
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals