#!/usr/bin/env python3
"""
6h_RSI50_Crossover_1wTrend_VolumeSpike
Hypothesis: Uses weekly trend (price above/below 50-period SMA) as directional filter, RSI(14) crossing above/below 50 as entry signal, and volume spike (>2x 20-period average) for confirmation on 6h timeframe. Designed to capture momentum shifts in both bull and bear markets by aligning with higher-timeframe trend while avoiding false signals in low-volume conditions. Targets 12-37 trades per year with discrete position sizing (0.25) to minimize fee churn.
"""

name = "6h_RSI50_Crossover_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for RSI and SMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_trend_up = close[i] > sma_50_1w_aligned[i]
        weekly_trend_down = close[i] < sma_50_1w_aligned[i]
        
        if position == 0:
            # Long: RSI crosses above 50, weekly uptrend, volume spike
            if (rsi[i] > 50 and rsi[i-1] <= 50 and 
                weekly_trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50, weekly downtrend, volume spike
            elif (rsi[i] < 50 and rsi[i-1] >= 50 and 
                  weekly_trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses below 50 or trend changes
            if (rsi[i] < 50 and rsi[i-1] >= 50) or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses above 50 or trend changes
            if (rsi[i] > 50 and rsi[i-1] <= 50) or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals