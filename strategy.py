#!/usr/bin/env python3
# 1d_Stochastic_Oversold_Bounce_1wTrend_Filter
# Hypothesis: Buy when daily stochastic falls below 20 (oversold) with weekly uptrend filter (price above 50-week SMA).
# Sell when stochastic rises above 80 (overbought) or weekly trend turns bearish.
# Uses mean reversion in oversold conditions with trend filter to avoid catching falling knives.
# Designed for 1d timeframe to achieve 7-25 trades/year with low frequency and high win rate.

name = "1d_Stochastic_Oversold_Bounce_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for stochastic
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Stochastic %K (14,3,3) on daily
    lookback = 14
    lowest_low = np.full_like(close_1d, np.nan)
    highest_high = np.full_like(close_1d, np.nan)
    for i in range(lookback - 1, len(close_1d)):
        lowest_low[i] = np.min(low_1d[i - lookback + 1:i + 1])
        highest_high[i] = np.max(high_1d[i - lookback + 1:i + 1])
    
    # Avoid division by zero
    diff = highest_high - lowest_low
    stoch_k = np.full_like(close_1d, np.nan)
    mask = diff != 0
    stoch_k[mask] = 100 * (close_1d[mask] - lowest_low[mask]) / diff[mask]
    
    # Stochastic %D (3-period SMA of %K)
    stoch_d = np.full_like(close_1d, np.nan)
    for i in range(2, len(stoch_k)):  # need 3 values for SMA
        if not np.isnan(stoch_k[i-2]) and not np.isnan(stoch_k[i-1]) and not np.isnan(stoch_k[i]):
            stoch_d[i] = (stoch_k[i-2] + stoch_k[i-1] + stoch_k[i]) / 3.0
    
    # Weekly 50 SMA for trend filter
    sma_50_1w = np.full_like(close_1w, np.nan)
    for i in range(49, len(close_1w)):  # 50-period SMA
        sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align all indicators to lower timeframe (wait for close of bar)
    stoch_d_aligned = align_htf_to_ltf(prices, df_1d, stoch_d)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(stoch_d_aligned[i]) or np.isnan(sma_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long when stochastic is oversold (<20) and weekly trend is bullish (price > 50-week SMA)
            if stoch_d_aligned[i] < 20 and close[i] > sma_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit when stochastic becomes overbought (>80) or weekly trend turns bearish (price < 50-week SMA)
            if stoch_d_aligned[i] > 80 or close[i] < sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals