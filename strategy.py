#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Confirmation
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies adaptive trend direction on daily timeframe.
Enter long when price > KAMA and volume above average; short when price < KAMA and volume above average.
Use weekly EMA200 as regime filter to avoid counter-trend trades in strong opposing weekly trends.
Discrete position sizing (0.25) minimizes fee churn. Target 15-25 trades/year to work in both bull and bear markets.
KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on daily close
    # Parameters: ER period=10, Fast SC=2, Slow SC=30
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series - close_1d_series.shift(10))
    volatility = abs(close_1d_series.diff()).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama_1d = [close_1d[0]]  # seed
    for i in range(1, len(close_1d)):
        kama_1d.append(kama_1d[-1] + sc.iloc[i] * (close_1d[i] - kama_1d[-1]))
    kama_1d = np.array(kama_1d)
    
    # Get weekly data for regime filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d, additional_delay_bars=1)
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w, additional_delay_bars=1)
    
    # Volume confirmation: 20-period average volume on 1d
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10) and volume MA (20)
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for entry signals with volume confirmation and weekly regime filter
            # Long: price > KAMA, volume above average, and weekly trend up (price > weekly EMA200)
            # Short: price < KAMA, volume above average, and weekly trend down (price < weekly EMA200)
            long_signal = (close[i] > kama_aligned[i]) and \
                         (volume[i] > volume_ma_20[i]) and \
                         (close[i] > ema200_aligned[i])
            short_signal = (close[i] < kama_aligned[i]) and \
                          (volume[i] > volume_ma_20[i]) and \
                          (close[i] < ema200_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below KAMA (trend reversal)
            exit_signal = close[i] < kama_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above KAMA (trend reversal)
            exit_signal = close[i] > kama_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0