#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend direction with 1d RSI filter and volume spike confirmation
# KAMA adapts to market noise, reducing whipsaw in choppy markets
# 1d RSI > 50 for long bias, < 50 for short bias ensures alignment with higher timeframe momentum
# Volume spike (>2x 20-period average) confirms momentum at breakout
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_KAMA_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d RSI for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on daily data
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h data
    def kama(data, period=10, fast=2, slow=30):
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(data, period))
        volatility = np.sum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        er = np.zeros_like(data)
        er[period:] = change[period-1:] / (volatility[period-1:] + 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Calculate KAMA
        kama_vals = np.zeros_like(data)
        kama_vals[0] = data[0]
        for i in range(1, len(data)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (data[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, period=10, fast=2, slow=30)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA + RSI > 50 (bullish bias) + volume spike
            if (close[i] > kama_vals[i] and 
                rsi_1d_aligned[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI < 50 (bearish bias) + volume spike
            elif (close[i] < kama_vals[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or RSI drops below 50
            if (close[i] < kama_vals[i]) or (rsi_1d_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or RSI rises above 50
            if (close[i] > kama_vals[i]) or (rsi_1d_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals