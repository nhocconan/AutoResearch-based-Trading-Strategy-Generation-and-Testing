#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d/1w EMA trend filter with price reversal signals at extreme Bollinger Bands.
# Long when price closes below lower Bollinger Band (20,2) with 1d EMA50 uptrend and 1w EMA200 uptrend.
# Short when price closes above upper Bollinger Band with 1d EMA50 downtrend and 1w EMA200 downtrend.
# Exit when price crosses the 20-period SMA (middle Bollinger Band).
# Uses Bollinger Bands for mean reversion in ranging markets, EMA filters for trend alignment.
# Target: 20-40 trades per year by requiring multi-timeframe trend alignment and extreme price action.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d and 1w data for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period_1d = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period_1d:
        ema_1d[ema_period_1d - 1] = np.mean(close_1d[:ema_period_1d])
        for i in range(ema_period_1d, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period_1d + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period_1d + 1))))
    
    # Calculate 1w EMA200 for trend filter
    ema_period_1w = 200
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period_1w:
        ema_1w[ema_period_1w - 1] = np.mean(close_1w[:ema_period_1w])
        for i in range(ema_period_1w, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period_1w + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period_1w + 1))))
    
    # Calculate Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std_dev = 2
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_middle = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
        bb_upper[i] = sma[i] + bb_std_dev * std_dev[i]
        bb_lower[i] = sma[i] - bb_std_dev * std_dev[i]
        bb_middle[i] = sma[i]
    
    # Align HTF EMAs to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Bollinger Bands, EMAs
    start_idx = max(bb_period - 1, ema_period_1d - 1, ema_period_1w - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price closes below lower BB with 1d EMA50 uptrend and 1w EMA200 uptrend
            if (price < bb_lower[i] and 
                price > ema_1d_aligned[i] and 
                price > ema_1w_aligned[i]):
                signals[i] = size
                position = 1
            # Short: price closes above upper BB with 1d EMA50 downtrend and 1w EMA200 downtrend
            elif (price > bb_upper[i] and 
                  price < ema_1d_aligned[i] and 
                  price < ema_1w_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above middle BB (SMA)
            if price > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses below middle BB (SMA)
            if price < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger20_2_Reversal_1dEMA50_1wEMA200"
timeframe = "4h"
leverage = 1.0