#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1-day KAMA trend filter with 1-day RSI mean reversion.
# Enters long when price dips below RSI(14) 30 with daily KAMA uptrend, short when price rises above RSI(14) 70 with daily KAMA downtrend.
# Uses 1-day timeframe for both KAMA trend and RSI to avoid look-ahead and ensure alignment.
# Designed to work in both bull and bear markets by fading extremes with trend filter.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_KAMA_RSI_MeanReversion_1d"
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
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for RSI and KAMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_1d_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) + price above KAMA (uptrend filter)
            if rsi_val < 30 and close[i] > kama_val:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought) + price below KAMA (downtrend filter)
            elif rsi_val > 70 and close[i] < kama_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or price below KAMA (trend change)
            if rsi_val > 50 or close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or price above KAMA (trend change)
            if rsi_val < 50 or close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals