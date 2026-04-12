#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_TrendFilter
Hypothesis: Use 1d KAMA direction (trend filter) and 4h RSI overbought/oversold levels for mean-reversion entries.
KAMA adapts to market noise, reducing whipsaw in ranging markets. RSI extremes provide entry points.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
Low trade frequency expected due to dual-condition requirement.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_TrendFilter"
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
    
    # === 1D KAMA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA parameters
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Handle edge cases for volatility calculation
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded != 0, change / volatility_padded, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[29] = close_1d[29]  # Start after enough data
    for i in range(30, len(close_1d)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 4H RSI FOR ENTRY SIGNALS ===
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Initial average gain/loss
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14]) if not np.any(np.isnan(gain[1:14])) else np.nan
    avg_loss[13] = np.mean(loss[1:14]) if not np.any(np.isnan(loss[1:14])) else np.nan
    
    # Wilder smoothing
    for i in range(14, len(close)):
        if not np.isnan(avg_gain[i-1]) and not np.isnan(avg_loss[i-1]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry signals
        long_signal = uptrend and rsi_oversold
        short_signal = downtrend and rsi_overbought
        
        # Exit on opposite RSI extreme or trend change
        exit_long = (position == 1 and 
                    (rsi[i] > 70 or not uptrend))
        exit_short = (position == -1 and 
                     (rsi[i] < 30 or not downtrend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals