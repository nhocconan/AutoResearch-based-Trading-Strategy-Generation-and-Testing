#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Bollinger Bands with KAMA trend filter and volume confirmation.
# Long when price touches lower BB with volume spike and KAMA trending up.
# Short when price touches upper BB with volume spike and KAMA trending down.
# Uses Bollinger Bands for mean reversion in ranging markets and KAMA for trend filtering.
# Designed for 20-40 trades/year to avoid fee drag, works in both bull and bear markets via mean reversion + trend filter.

name = "4h_1dBB_KAMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    sma_20 = pd.Series(close_1d).rolling(window=bb_length, min_periods=bb_length).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_bb = sma_20 + (bb_mult * std_20)
    lower_bb = sma_20 - (bb_mult * std_20)
    
    # Calculate 1d KAMA (ER=10, fast=2, slow=30)
    er_length = 10
    fast_sc = 2
    slow_sc = 30
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d volume SMA for spike detection
    vol_sma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    vol_sma_aligned = align_htf_to_ltf(prices, df_1d, vol_sma)
    
    # 4h volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(vol_sma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches lower BB + volume spike + price > KAMA (uptrend)
            if low[i] <= lower_bb_aligned[i] and vol_spike[i] and close[i] > kama_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper BB + volume spike + price < KAMA (downtrend)
            elif high[i] >= upper_bb_aligned[i] and vol_spike[i] and close[i] < kama_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above KAMA or touches upper BB
            if close[i] >= kama_aligned[i] or high[i] >= upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below KAMA or touches lower BB
            if close[i] <= kama_aligned[i] or low[i] <= lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals