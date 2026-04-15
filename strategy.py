#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA (Kaufman Adaptive Moving Average) trend + RSI pullback + volume confirmation
# Uses KAMA for adaptive trend detection (adapts to market noise), RSI(14) for pullback entries in trend direction,
# and volume spike to confirm momentum. Works in both bull and bear by only taking long in uptrend (price > KAMA)
# and short in downtrend (price < KAMA). Target: 60-120 total trades over 4 years (15-30/year) with high-quality entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for trend and price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close_4h - np.roll(close_4h, 10))
    change[0:10] = 0  # First 10 values
    volatility = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    
    # Smoothing constants: SC = [ER * (fastest - slowest) + slowest]^2
    # fastest = 2/(2+1) = 0.6667, slowest = 2/(30+1) = 0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_4h, np.nan)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14-period) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price > KAMA (uptrend) + RSI < 40 (pullback) + volume spike
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] < 40 and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price < KAMA (downtrend) + RSI > 60 (pullback) + volume spike
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] > 60 and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price crosses KAMA (trend change)
        elif position == 1 and close[i] < kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > kama_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSI_Pullback_Volume"
timeframe = "4h"
leverage = 1.0