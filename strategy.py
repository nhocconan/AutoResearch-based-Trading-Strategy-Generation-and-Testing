#!/usr/bin/env python3
# 6H_RSI_Divergence_1DTrend_Filter
# Hypothesis: 6-hour strategy combining RSI divergence (bullish/bearish) with 1-day EMA50 trend filter and volume confirmation.
# RSI divergence identifies potential reversals, while the 1-day EMA50 filter ensures trades align with the daily trend.
# Volume confirmation reduces false signals. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear markets by using trend filter to avoid counter-trend trades.

name = "6H_RSI_Divergence_1DTrend_Filter"
timeframe = "6h"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for RSI and EMA
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate RSI (14-period) on 6-hour data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate RSI divergence signals (lookback 5 periods)
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    rsi_bull_div = np.zeros(n, dtype=bool)
    rsi_bear_div = np.zeros(n, dtype=bool)
    
    lookback = 5
    for i in range(lookback, n):
        # Bullish divergence: lower price low, higher RSI low
        if (low[i] < low[i-lookback] and 
            rsi[i] > rsi[i-lookback]):
            # Check if this is a meaningful low point
            if low[i] == np.min(low[i-lookback:i+1]):
                rsi_bull_div[i] = True
        
        # Bearish divergence: higher price high, lower RSI high
        if (high[i] > high[i-lookback] and 
            rsi[i] < rsi[i-lookback]):
            # Check if this is a meaningful high point
            if high[i] == np.max(high[i-lookback:i+1]):
                rsi_bear_div[i] = True
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: confirmation (1.5x average volume)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish RSI divergence + uptrend (close > EMA50) + volume confirmation
            if (rsi_bull_div[i] and 
                close[i] > ema_50_aligned[i] and   # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bearish RSI divergence + downtrend (close < EMA50) + volume confirmation
            elif (rsi_bear_div[i] and 
                  close[i] < ema_50_aligned[i] and   # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Opposite RSI divergence signal (trend exhaustion)
            # 2. Price crosses EMA50 (trend change)
            opposite_div = (position == 1 and rsi_bear_div[i]) or \
                           (position == -1 and rsi_bull_div[i])
            ema_cross = (position == 1 and close[i] < ema_50_aligned[i]) or \
                        (position == -1 and close[i] > ema_50_aligned[i])
            
            if opposite_div or ema_cross:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals