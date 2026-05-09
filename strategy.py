#!/usr/bin/env python3
# Hypothesis: 4h KAMA trend with Bollinger Band mean reversion and volume confirmation
# Long when price touches lower Bollinger Band in KAMA uptrend with volume > 1.5x average
# Short when price touches upper Bollinger Band in KAMA downtrend with volume > 1.5x average
# Exit when price crosses KAMA or opposite Bollinger Band
# Uses KAMA for adaptive trend, Bollinger Bands for mean reversion, volume for conviction
# Designed to work in both trending and ranging markets with controlled frequency
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_KAMA_BB_MeanRev_Volume"
timeframe = "4h"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio and KAMA
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # Simplified for calculation
    
    # Proper KAMA calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i >= 10:  # Lookback period for ER
            net_change = abs(close_1d[i] - close_1d[i-10])
            total_change = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            if total_change > 0:
                er[i] = net_change / total_change
            else:
                er[i] = 0
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate Bollinger Bands on 1d timeframe (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches lower BB in KAMA uptrend with volume spike
            if (low[i] <= lower_bb_aligned[i] and 
                kama_aligned[i] > kama_aligned[i-1] and  # KAMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper BB in KAMA downtrend with volume spike
            elif (high[i] >= upper_bb_aligned[i] and 
                  kama_aligned[i] < kama_aligned[i-1] and  # KAMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses KAMA or touches upper BB
            if (close[i] >= kama_aligned[i]) or (high[i] >= upper_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses KAMA or touches lower BB
            if (close[i] <= kama_aligned[i]) or (low[i] <= lower_bb_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals