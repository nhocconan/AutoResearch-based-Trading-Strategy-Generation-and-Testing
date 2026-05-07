#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_v2
Hypothesis: On 1d timeframe, use KAMA to determine trend direction (trending vs ranging), RSI for momentum confirmation, and Choppiness Index as regime filter. Enter long when KAMA is trending upward, RSI > 50, and market is trending (CHOP < 38.2); enter short when KAMA is trending downward, RSI < 50, and market is trending. Uses weekly trend filter to avoid counter-trend trades. Designed for low frequency (15-25 trades/year) to minimize fee drift while capturing sustained moves in both bull and bear markets.
"""
name = "1d_KAMA_Direction_RSI_Chop_v2"
timeframe = "1d"
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
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close, k=10))  # |close - close 10 periods ago|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes over 10 periods
    # Handle first 9 values where diff doesn't exist
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate SSC (Smoothing Constant)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[kama_period-1] = close[kama_period-1]  # seed
    for i in range(kama_period, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    # Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_high - min_low
    chop = np.where(atr > 0, 100 * np.log10(range_14 / (atr * 14)) / np.log10(14), 50)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, kama_period)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA trending up (close > KAMA), RSI > 50, trending market (CHOP < 38.2), weekly uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 38.2 and 
                close[i] > ema50_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down (close < KAMA), RSI < 50, trending market (CHOP < 38.2), weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 38.2 and 
                  close[i] < ema50_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: opposite condition
            if position == 1:
                if close[i] < kama[i] or rsi[i] < 40:  # exit on trend change or RSI drop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or rsi[i] > 60:  # exit on trend change or RSI rise
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals