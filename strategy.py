#!/usr/bin/env python3
"""
#100870 - 1d_KAMA_20_Trend_RSI2_MeanRev_1wTrend
Hypothesis: On daily timeframe, use KAMA(20) for trend direction, RSI(2) for mean-reversion entries, and weekly EMA(20) for trend filter. Long when KAMA up, RSI<10, close>weekly EMA; short when KAMA down, RSI>90, close<weekly EMA. Exit on opposite RSI extreme or trend change. Designed for low trade frequency (<15/year) to avoid fee drag, works in bull (trend follow) and bear (mean reversion to trend).
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
    
    # KAMA(20) for trend direction
    # Efficiency ratio: |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # Not enough data for first 10 periods
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.nansum(np.abs(close[i-9:i+1] - np.roll(close[i-9:i+1], 1)))
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(2) for mean-reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine KAMA trend: up if current > previous, down if current < previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # Long condition: KAMA up, RSI < 10 (oversold), close > weekly EMA20
        if (kama_up and rsi[i] < 10 and close[i] > ema20_1w_aligned[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: KAMA down, RSI > 90 (overbought), close < weekly EMA20
        elif (kama_down and rsi[i] > 90 and close[i] < ema20_1w_aligned[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI reaches opposite extreme or trend change
        elif position == 1 and (rsi[i] > 90 or not kama_up):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi[i] < 10 or not kama_down):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_20_Trend_RSI2_MeanRev_1wTrend"
timeframe = "1d"
leverage = 1.0