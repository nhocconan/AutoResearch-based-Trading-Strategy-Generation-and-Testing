#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_KAMA_Crossover_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly KAMA trend filter
    close_1w = df_1w['close'].values
    # KAMA parameters
    close_s = pd.Series(close_1w)
    change = np.abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    trend_1w = (close_1w > kama_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # KAMA crossover on 6h
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(2))
    volatility = close_s.diff().abs().rolling(window=2, min_periods=2).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    
    # Crossover signals
    bullish_cross = (close > kama) & (close <= kama_prev)
    bearish_cross = (close < kama) & (close >= kama_prev)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for KAMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(trend_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(kama_prev[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: bullish crossover with weekly uptrend
            if bullish_cross[i] and trend_1w_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish crossover with weekly downtrend
            elif bearish_cross[i] and trend_1w_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish crossover
            if bearish_cross[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish crossover
            if bullish_cross[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA crossover signals filtered by weekly KAMA trend.
# In bull markets, weekly trend filter allows long entries on 6h bullish crossovers.
# In bear markets, weekly trend filter allows short entries on 6h bearish crossovers.
# KAMA adapts to market conditions, reducing whipsaws in ranging markets.
# Target: 15-30 trades/year to avoid fee drag while capturing trend changes.