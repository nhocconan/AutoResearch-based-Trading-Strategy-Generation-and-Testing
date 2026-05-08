#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend with 1-day RSI and volume confirmation
# We go long when KAMA indicates uptrend, daily RSI > 50 (bullish momentum), and volume spike.
# We go short when KAMA indicates downtrend, daily RSI < 50 (bearish momentum), and volume spike.
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# KAMA adapts to market noise, reducing false signals in choppy conditions.
# Daily RSI filters for momentum alignment with higher timeframe.
# Volume spike confirms institutional participation.

name = "12h_KAMA_RSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for RSI and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Calculate KAMA on 12h prices
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder - will compute properly below
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            change_val = np.abs(close[i] - close[i-10])
            volatility_val = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if volatility_val > 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 0
    # Smoothing constants
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike: current volume > 1.5 * 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_1d_aligned[i]
        kama_val = kama[i]
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) + RSI > 50 + volume spike
            if close_val > kama_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) + RSI < 50 + volume spike
            elif close_val < kama_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA OR RSI < 50
            if close_val < kama_val or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA OR RSI > 50
            if close_val > kama_val or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals