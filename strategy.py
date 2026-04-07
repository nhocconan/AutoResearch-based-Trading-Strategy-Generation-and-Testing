#!/usr/bin/env python3
"""
4h_kama_rsi_chop_filter_v1
Hypothesis: KAMA (Kaufman Adaptive Moving Average) trend direction on 4h combined with RSI momentum and Choppiness Index regime filter.
Long when KAMA turns up, RSI > 50, and market is trending (CHOP < 38.2). Short when KAMA turns down, RSI < 50, and market is trending.
Designed for 20-30 trades/year on 4h timeframe with adaptive trend following that works in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_rsi_chop_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([np.full(10, np.nan), er])
    
    # Calculate smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14), 
                    50)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI conditions
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        
        # Chop regime: trending when CHOP < 38.2
        chop_trending = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: KAMA turns down or chop becomes ranging
            if kama_down or not chop_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up or chop becomes ranging
            if kama_up or not chop_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: KAMA turning up, RSI > 50, and trending market
            if kama_up and rsi_above_50 and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short: KAMA turning down, RSI < 50, and trending market
            elif kama_down and rsi_below_50 and chop_trending:
                position = -1
                signals[i] = -0.25
    
    return signals