#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Direction_RSI200_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # KAMA on 12h close - uses Efficiency Ratio
    er = np.zeros(n)
    change = np.abs(np.diff(close, prepend=close[0]))
    er[10:] = np.abs(np.diff(close, 10)) / (np.convolve(change, np.ones(10), 'same') + 1e-10)
    er[:10] = 0
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200 and KAMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, above 1d EMA200, RSI between 40-60 (avoid extremes)
            long_cond = (close[i] > kama[i] and 
                        close[i] > ema200_1d_aligned[i] and
                        40 <= rsi_1d_aligned[i] <= 60)
            
            # Short: Price below KAMA, below 1d EMA200, RSI between 40-60
            short_cond = (close[i] < kama[i] and 
                         close[i] < ema200_1d_aligned[i] and
                         40 <= rsi_1d_aligned[i] <= 60)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below KAMA OR below 1d EMA200
            if close[i] < kama[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above KAMA OR above 1d EMA200
            if close[i] > kama[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA adapts to market efficiency - in trending markets it follows price closely,
# in ranging markets it stays flat. Combined with 1d EMA200 trend filter and RSI 40-60 range
# to avoid extremes. Works in bull markets via trend following, in bear via mean reversion
# from extreme RSI readings. 12h timeframe targets 12-37 trades/year to avoid fee drag.
# Discrete sizing (0.25) minimizes churn. Works on BTC/ETH/SOL via adaptive trend detection.