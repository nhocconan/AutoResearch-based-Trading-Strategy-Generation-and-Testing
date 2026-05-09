#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily KAMA trend with 1w EMA200 filter and volume confirmation.
# Uses KAMA (Kaufman Adaptive Moving Average) to adapt to market conditions - faster in trends, slower in ranges.
# 1-week EMA200 ensures alignment with higher timeframe trend for multi-timeframe confirmation.
# Volume confirmation filters out low-conviction moves.
# Designed to work in both bull and bear markets by following adaptive trend.
name = "1d_KAMA_Trend_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Proper ER calculation: |close - close[n-10]| / sum(|close[i] - close[i-1]|) over 10 periods
    lookback = 10
    net_change = np.abs(np.subtract(close[lookback:], close[:-lookback]) if len(close) > lookback else np.array([]))
    total_change = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Vectorized ER calculation
    er = np.zeros_like(close)
    for i in range(lookback, len(close)):
        net_ch = np.abs(close[i] - close[i-lookback])
        tot_ch = np.sum(np.abs(np.diff(close[i-lookback:i+1])))
        er[i] = net_ch / tot_ch if tot_ch != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly EMA200 trend filter
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirmed = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 200)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(kama[i]) or np.isnan(ema_200_1d[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above KAMA and above weekly EMA200 with volume confirmation
            if (price > kama[i] and price > ema_200_1d[i] and vol_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below weekly EMA200 with volume confirmation
            elif (price < kama[i] and price < ema_200_1d[i] and vol_confirmed[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below KAMA
            if price < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above KAMA
            if price > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals