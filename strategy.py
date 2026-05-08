#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_RSI_Overbought_Oversold"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(30) for trend direction
    close_1w = df_1w['close'].values
    ema30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema30_1w)
    
    # Calculate KAMA on daily data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |diff| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema30_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema30_1w_val = ema30_1w_aligned[i]
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) + RSI < 30 (oversold) + weekly uptrend
            if (close[i] > kama_val and 
                rsi_val < 30 and 
                close[i] > ema30_1w_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) + RSI > 70 (overbought) + weekly downtrend
            elif (close[i] < kama_val and 
                  rsi_val > 70 and 
                  close[i] < ema30_1w_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 70 (overbought)
            if (close[i] < kama_val or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 30 (oversold)
            if (close[i] > kama_val or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals