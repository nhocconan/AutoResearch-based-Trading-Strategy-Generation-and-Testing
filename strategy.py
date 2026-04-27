#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Momentum
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 1d timeframe with weekly trend filter and RSI momentum confirmation to capture sustained trends in both bull and bear markets. Targets 15-25 trades/year on 1d to minimize fee drag while maintaining high win rate. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false signals during choppy periods.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate KAMA on daily data
    # KAMA parameters: ER period=10, fast SC=2, slow SC=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), 
                      axis=0 if len(change.shape) == 1 else None, keepdims=True)
    # For 1D array, calculate rolling volatility
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    change_abs = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    # Avoid division by zero
    er = np.where(volatility != 0, change_abs / volatility, 0)
    sc = (er * (2/10 - 1/30) + 1/30) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        ema_trend = ema20_1w_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: price above KAMA with weekly uptrend and RSI > 50
            if close[i] > kama_val and ema_trend > kama_val and rsi_val > 50:
                signals[i] = size
                position = 1
            # Short: price below KAMA with weekly downtrend and RSI < 50
            elif close[i] < kama_val and ema_trend < kama_val and rsi_val < 50:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA or weekly trend turns down
            if close[i] < kama_val or ema_trend < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above KAMA or weekly trend turns up
            if close[i] > kama_val or ema_trend > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Momentum"
timeframe = "1d"
leverage = 1.0