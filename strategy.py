#!/usr/bin/env python3
# 4h_12h_KAMA_RSI_Trend_With_Volume_Filter
# Hypothesis: 4h signals based on KAMA direction (trend) from 12h, confirmed by RSI extremes and volume spikes.
# Uses 12h KAMA to determine trend direction, enters long when RSI < 30 and short when RSI > 70, only in the direction of the 12h trend.
# Volume must be > 2x 20-period average to confirm momentum.
# Designed to work in both bull and bear markets by following the 12h trend and using mean-reversion entries within the trend.
# Expected trade count: ~15-25 per year per symbol to avoid fee drag.

name = "4h_12h_KAMA_RSI_Trend_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for KAMA and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h KAMA (trend indicator)
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will compute correctly below
    # Recompute volatility properly: sum of absolute changes over window
    volatility = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_12h[max(0, i-9):i+1])))  # 10-period ER
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 12h RSI (14-period) for mean-reversion entries
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume confirmation (20-period for 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_12h_aligned[i]) or
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h KAMA: price > KAMA = uptrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > kama_12h_aligned[i]
        downtrend = close_12h_aligned[i] < kama_12h_aligned[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) in uptrend with volume
            if rsi_12h_aligned[i] < 30 and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) in downtrend with volume
            elif rsi_12h_aligned[i] > 70 and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: RSI > 50 (mean reversion) or trend fails
                if rsi_12h_aligned[i] > 50 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: RSI < 50 (mean reversion) or trend fails
                if rsi_12h_aligned[i] < 50 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals