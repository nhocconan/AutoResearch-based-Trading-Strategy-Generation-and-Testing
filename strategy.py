#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Trend_Filter_v3
Hypothesis: On daily timeframe, use KAMA to determine trend direction, RSI(2) for mean-reversion entries within the trend, and weekly trend filter to avoid counter-trend trades. KAMA adapts to market noise, reducing false signals in chop. RSI(2) captures short-term reversals in trending markets. Weekly trend ensures alignment with higher-timeframe momentum. Designed for low trade frequency (<20/year) to minimize fee drag in both bull and bear markets.
"""

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
    
    # KAMA trend indicator on daily
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.subtract(close_1d[1:], close_1d[:-1]))
    volatility = np.append(volatility, 0)  # same length
    er = np.zeros_like(close_1d, dtype=np.float64)
    for i in range(len(close_1d)):
        if i >= 9:
            direction = np.abs(close_1d[i] - close_1d[i-9])
            volatility_sum = np.sum(volatility[i-9:i+1])
            er[i] = direction / volatility_sum if volatility_sum != 0 else 0
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(2) on daily for entry signals
    rsi_period = 2
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Weekly trend filter: price above/below weekly EMA20
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # start after warmup
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_20_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Long: in uptrend, RSI(2) oversold (<10), price above weekly EMA20
        long_condition = uptrend and (rsi_aligned[i] < 10) and (close[i] > ema_20_weekly_aligned[i])
        
        # Short: in downtrend, RSI(2) overbought (>90), price below weekly EMA20
        short_condition = downtrend and (rsi_aligned[i] > 90) and (close[i] < ema_20_weekly_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_KAMA_RSI_Trend_Filter_v3"
timeframe = "1d"
leverage = 1.0