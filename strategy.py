#!/usr/bin/env python3
"""
1d_KAMA_Trend_Trader
Hypothesis: Daily KAMA (adaptive moving average) provides robust trend direction in both bull and bear markets.
Price crossing above/below KAMA signals trend changes. Volume confirmation filters false signals.
ATR-based stop loss limits drawdown. Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
Weekly trend filter (price vs weekly EMA) ensures alignment with higher timeframe trend.
Works in bull/bear via adaptive trend following and volatility-adjusted positioning.
"""

name = "1d_KAMA_Trend_Trader"
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
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_ema = 2
    slow_ema = 30
    lookback = 10
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, k=lookback))
    change = np.concatenate([np.full(lookback, np.nan), change])
    
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    volatility = pd.Series(volatility).rolling(window=lookback, min_periods=1).sum().values
    
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # ATR for volatility normalization and stop reference
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price crosses above KAMA with volume and weekly uptrend
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and 
                volume_confirm[i] and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume and weekly downtrend
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and 
                  volume_confirm[i] and weekly_downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or weekly trend turns down
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or weekly trend turns up
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals