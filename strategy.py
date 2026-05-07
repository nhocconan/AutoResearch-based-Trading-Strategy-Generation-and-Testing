#!/usr/bin/env python3
# 4h_RSI_Stochastic_Confluence_Trend
# Hypothesis: On 4h chart, enter long when RSI < 40 (oversold) AND %K crosses above %D (momentum up) AND price > 200-period SMA (long-term trend).
# Enter short when RSI > 60 (overbought) AND %K crosses below %D (momentum down) AND price < 200-period SMA.
# Use RSI and Stochastic for mean-reversion entries with trend filter to avoid counter-trend trades.
# Designed for low trade frequency (~20-40/year) to minimize fee drag and work in trending markets.
# Works in both bull and bear markets by only taking trades in direction of long-term trend.
timeframe = "4h"
name = "4h_RSI_Stochastic_Confluence_Trend"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 200-period SMA for trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Stochastic crossover signals
    k_cross_above_d = (k_percent > d_percent) & (np.roll(k_percent, 1) <= np.roll(d_percent, 1))
    k_cross_below_d = (k_percent < d_percent) & (np.roll(k_percent, 1) >= np.roll(d_percent, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any critical value is NaN
        if (np.isnan(sma_200[i]) or np.isnan(rsi[i]) or np.isnan(k_percent[i]) or 
            np.isnan(d_percent[i]) or np.isnan(k_cross_above_d[i]) or np.isnan(k_cross_below_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 40 (oversold) AND %K crosses above %D AND price > 200 SMA
            if rsi[i] < 40 and k_cross_above_d[i] and close[i] > sma_200[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60 (overbought) AND %K crosses below %D AND price < 200 SMA
            elif rsi[i] > 60 and k_cross_below_d[i] and close[i] < sma_200[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 60 (overbought) OR price < 200 SMA (trend change)
            if rsi[i] > 60 or close[i] < sma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 40 (oversold) OR price > 200 SMA (trend change)
            if rsi[i] < 40 or close[i] > sma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals