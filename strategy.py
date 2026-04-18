#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1-day trend filter (SMA50) and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (price > SMA50),
# we look for pullbacks to enter in direction of trend. Volume confirms momentum.
# Designed for 12h timeframe to limit trades (target 20-50/year) and avoid fee drag.
# Works in both bull and bear markets by using trend filter to only trade with higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (SMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily SMA(50) for trend filter
    sma_50_1d = np.full(len(close_1d), np.nan)
    for i in range(50, len(close_1d)):
        sma_50_1d[i] = np.mean(close_1d[i-50:i])
    
    # Align daily SMA50 to 12h timeframe
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate Williams %R(14) on 12h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i])
        lowest_low[i] = np.min(low[i-14:i])
    # Initialize first 14 values
    for i in range(14):
        highest_high[i] = np.max(high[:i+1])
        lowest_low[i] = np.min(low[:i+1])
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r[highest_high == lowest_low] = -50
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # need daily SMA50, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price above daily SMA50 (uptrend) or below (downtrend)
        trend_up = close[i] > sma_50_1d_aligned[i]
        trend_down = close[i] < sma_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) in uptrend with volume
            if (williams_r[i] < -80 and 
                trend_up and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) in downtrend with volume
            elif (williams_r[i] > -20 and 
                  trend_down and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or trend reversal
            if williams_r[i] > -20 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or trend reversal
            if williams_r[i] < -80 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_SMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0