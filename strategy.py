#!/usr/bin/env python3
# 12h_1d_KAMA_Momentum_With_Volume_Confirmation
# Hypothesis: On 12h timeframe, use KAMA (Kaufman Adaptive Moving Average) to detect momentum direction,
# combined with volume confirmation and a 1d RSI filter. In both bull and bear markets,
# momentum persists when volume supports it, and RSI helps avoid overextended entries.
# The 1d RSI filter prevents entries during extreme overbought/oversold conditions on the daily chart.
# Targets 15-30 trades/year by requiring alignment of 12h momentum, volume surge, and 1d RSI not extreme.

name = "12h_1d_KAMA_Momentum_With_Volume_Confirmation"
timeframe = "12h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for overbought/oversold filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    def rsi_wilder(series, period):
        result = np.full_like(series, np.nan)
        if len(series) < period:
            return result
        avg_gain = np.nansum(gain[:period]) / period
        avg_loss = np.nansum(loss[:period]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        result[period-1] = 100 - (100 / (1 + rs))
        for i in range(period, len(series)):
            avg_gain = (avg_gain * (period-1) + gain[i-1]) / period
            avg_loss = (avg_loss * (period-1) + loss[i-1]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
            result[i] = 100 - (100 / (1 + rs))
        return result
    
    rsi_1d = rsi_wilder(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate KAMA on 12h close
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # KAMA calculation
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_values = kama(close, 10, 2, 30)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_values[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, volume confirmation, RSI not overbought
            if (close[i] > kama_values[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                rsi_1d_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, volume confirmation, RSI not oversold
            elif (close[i] < kama_values[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  rsi_1d_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or RSI overbought
            if close[i] < kama_values[i] or rsi_1d_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or RSI oversold
            if close[i] > kama_values[i] or rsi_1d_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals