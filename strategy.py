#!/usr/bin/env python3
# 4h_1d_kama_volume_cross_v1
# Strategy: 4h KAMA crossover with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both bull and bear markets. 
# Price crossing above/below KAMA indicates trend changes. Volume confirms institutional participation.
# 1d EMA filter ensures we only trade in the direction of the higher timeframe trend, reducing false signals.
# Discrete position sizing (0.25) minimizes fee churn. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_volume_cross_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(close)):
        if i == 1:
            change[i] = np.abs(close[i] - close[i-1])
            volatility[i] = np.abs(close[i] - close[i-1])
        else:
            change[i] = np.abs(close[i] - close[i-10]) if i >= 10 else np.abs(close[i] - close[0])
            volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-9):i+1])))
    
    # Vectorized ER calculation
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros_like(close)
    for i in range(len(close)):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-9):i+1])))
    
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        # Long: Price crosses above KAMA + above 1d EMA50 (uptrend) + volume confirmation
        if vol_confirmed and close[i] > kama[i] and close[i-1] <= kama[i-1] and close[i] > ema_50_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price crosses below KAMA + below 1d EMA50 (downtrend) + volume confirmation
        elif vol_confirmed and close[i] < kama[i] and close[i-1] >= kama[i-1] and close[i] < ema_50_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to KAMA or trend reversal
        elif position == 1 and (close[i] < kama[i] or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama[i] or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals