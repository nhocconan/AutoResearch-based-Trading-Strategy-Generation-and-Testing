#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Chop_Filter
Hypothesis: Trade 12h KAMA trend direction with RSI momentum filter and choppiness regime filter to avoid whipsaws. 
KAMA adapts to market efficiency, reducing lag in trends and whipsaws in ranges. RSI filters for momentum strength. 
Choppiness index > 61.8 avoids ranging markets where trend following fails. 
Position size 0.25 balances profit and fee drag. Target: 12-25 trades/year (~50-100 over 4 years).
Works in bull/bear: KAMA follows trend, RSI confirms momentum, chop filter avoids false signals in ranges.
"""

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
    
    # Get 1d data for chop filter and volume MA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (12h close)
    # Efficiency ratio: |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))  # length n-10
    volatility = np.sum(np.abs(np.subtract(close[1:], close[:-1])), axis=0)  # length n-1
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility, np.full(9, np.nan)])  # align to close
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after first 10 bars
    for i in range(10, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14) from 1d data
    # True Range
    tr1 = np.subtract(high_1d[1:], low_1d[1:])
    tr2 = np.abs(np.subtract(high_1d[1:], close_1d[:-1]))
    tr3 = np.abs(np.subtract(low_1d[1:], close_1d[:-1]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align to 1d index
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sumTR14 / (maxHH - minLL)) / log10(14)
    chop = np.where((max_high_14 - min_low_14) != 0,
                    100 * np.log10(tr_sum_14 / (max_high_14 - min_low_14)) / np.log10(14),
                    0)
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), RSI (14), volume MA (20)
    start_idx = max(10, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (Chop < 61.8)
        trending_market = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price > KAMA AND RSI > 50 (bullish momentum) AND trending market AND volume confirm
            long_setup = (close[i] > kama[i]) and \
                         (rsi[i] > 50) and \
                         trending_market and \
                         volume_confirm[i]
            # Short: price < KAMA AND RSI < 50 (bearish momentum) AND trending market AND volume confirm
            short_setup = (close[i] < kama[i]) and \
                          (rsi[i] < 50) and \
                          trending_market and \
                          volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 40 (loss of momentum) OR chop > 61.8 (ranging market)
            if (close[i] < kama[i]) or \
               (rsi[i] < 40) or \
               (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60 (loss of momentum) OR chop > 61.8 (ranging market)
            if (close[i] > kama[i]) or \
               (rsi[i] > 60) or \
               (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0