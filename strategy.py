#!/usr/bin/env python3
# 1d_1w_RSI_Divergence_TrendFilter_v1
# Hypothesis: On 1d timeframe, trade RSI divergence with 1w trend filter.
# Bullish divergence (price makes lower low, RSI makes higher low) in uptrend (price > 1w EMA200) -> long.
# Bearish divergence (price makes higher high, RSI makes lower high) in downtrend (price < 1w EMA200) -> short.
# Uses volume confirmation to avoid false signals. Targets 10-25 trades/year by requiring confluence of
# divergence, trend, and volume. Designed to work in both bull and bear markets by following higher timeframe trend.

name = "1d_1w_RSI_Divergence_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i-2]) or np.isnan(rsi[i-1]) or np.isnan(rsi[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-1] and low[i-1] < low[i-2] and  # Lower low in price
                rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2] and  # Higher low in RSI
                close[i] > ema200_1w_aligned[i] and           # Uptrend filter (price > 1w EMA200)
                volume[i] > 1.5 * volume_ma[i]):              # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (high[i] > high[i-1] and high[i-1] > high[i-2] and  # Higher high in price
                  rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2] and      # Lower high in RSI
                  close[i] < ema200_1w_aligned[i] and                # Downtrend filter (price < 1w EMA200)
                  volume[i] > 1.5 * volume_ma[i]):                   # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence or price crosses below 1w EMA200
            if ((high[i] > high[i-1] and high[i-1] > high[i-2] and
                 rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]) or    # Bearish divergence
                close[i] < ema200_1w_aligned[i]):                 # Trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence or price crosses above 1w EMA200
            if ((low[i] < low[i-1] and low[i-1] < low[i-2] and
                 rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]) or    # Bullish divergence
                close[i] > ema200_1w_aligned[i]):                 # Trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals