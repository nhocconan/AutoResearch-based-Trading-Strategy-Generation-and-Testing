#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h KAMA (Kaufman Adaptive Moving Average) + 12h RSI + volume confirmation
# KAMA adapts to market noise - slow in ranging markets, fast in trending markets
# 12h RSI provides higher timeframe momentum filter
# Volume confirmation ensures breakouts have conviction
# Designed to work in both bull and bear markets by adapting speed to market conditions
# Target: 15-25 trades/year per symbol with disciplined entries
name = "6h_KAMA_12hRSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h RSI for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    rsi_period = 14
    delta = np.diff(df_12h['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(df_12h['close'], np.nan, dtype=float)
    avg_loss = np.full_like(df_12h['close'], np.nan, dtype=float)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # KAMA on 6h data
    def kama(data, period=10, fast=2, slow=30):
        kama_values = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return kama_values
        
        # Efficiency ratio
        change = np.abs(np.diff(data, period))
        volatility = np.sum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        er = np.zeros_like(data)
        for i in range(period, len(data)):
            if volatility[i-period+1:i+1].sum() > 0:
                er[i] = change[i] / volatility[i-period+1:i+1].sum()
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Initialize KAMA
        kama_values[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            kama_values[i] = kama_values[i-1] + sc[i] * (data[i] - kama_values[i-1])
        
        return kama_values
    
    kama_values = kama(close, period=10, fast=2, slow=30)
    
    # Volume spike: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_values[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA + RSI > 50 (bullish momentum) + volume spike
            if (close[i] > kama_values[i] and 
                rsi_12h_aligned[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI < 50 (bearish momentum) + volume spike
            elif (close[i] < kama_values[i] and 
                  rsi_12h_aligned[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or RSI turns bearish
            if (close[i] < kama_values[i]) or (rsi_12h_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or RSI turns bullish
            if (close[i] > kama_values[i]) or (rsi_12h_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals