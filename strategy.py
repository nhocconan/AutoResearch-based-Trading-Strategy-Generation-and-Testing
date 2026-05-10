#!/usr/bin/env python3
# 6h_RSI_Weakness_WeeklyTrend_Follow
# Hypothesis: In strong weekly trends (price above/below weekly 200 EMA), buy RSI weakness (pullbacks) on 6h.
# In bull regime (price > weekly 200 EMA), go long when RSI(14) < 40. In bear regime (price < weekly 200 EMA),
# go short when RSI(14) > 60. Uses weekly trend filter to avoid counter-trend trades.
# Low turnover expected due to weekly filter + RSI extreme conditions.
# Target: 10-25 trades/year on 6h timeframe.

name = "6h_RSI_Weakness_WeeklyTrend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Calculate Relative Strength Index"""
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly 200 EMA for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get 6h data for RSI calculation
    close = prices['close'].values
    
    # Calculate RSI on 6h timeframe
    rsi_6h = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA200 (200) + RSI (14)
    start_idx = max(200, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime based on weekly trend
        price_vs_weekly_ema = close[i] > ema200_1w_aligned[i]  # True = bull regime, False = bear regime
        
        if position == 0:
            # Long signal: bull regime + RSI oversold (< 40)
            if price_vs_weekly_ema and rsi_6h[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short signal: bear regime + RSI overbought (> 60)
            elif not price_vs_weekly_ema and rsi_6h[i] > 60:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought (> 70) or trend change (price < weekly EMA)
            if rsi_6h[i] > 70 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold (< 30) or trend change (price > weekly EMA)
            if rsi_6h[i] < 30 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals