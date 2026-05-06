#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily KAMA direction with RSI filter and volume confirmation
# - Uses 1d KAMA (adaptive moving average) to capture trend direction
# - Uses 12h RSI(14) for momentum confirmation (RSI > 50 for long, < 50 for short)
# - Uses 12h volume spike (2x 20-period average) for entry confirmation
# - Exits when KAMA direction reverses or RSI crosses back through 50
# - Designed to work in both bull and bear markets by following adaptive trend
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dKAMA_12hRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA (adaptive moving average)
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0]))
        
        # ER (Efficiency Ratio)
        er = np.zeros_like(close)
        for i in range(length, len(close)):
            abs_change = np.sum(np.abs(np.diff(close[i-length+1:i+1])))
            total_change = np.sum(np.abs(np.diff(close[i-length+1:i+1])))
            if total_change != 0:
                er[i] = abs_change / total_change
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        
        return kama_vals
    
    kama_1d = kama(close_1d, 10, 2, 30)
    kama_1d_dir = np.where(kama_1d > np.roll(kama_1d, 1), 1, -1)  # 1 for up, -1 for down
    kama_1d_dir[0] = 1  # Initialize
    
    # Align 1d KAMA direction to 12h timeframe
    kama_dir_12h = align_htf_to_ltf(prices, df_1d, kama_1d_dir)
    
    # RSI filter (12h timeframe)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Initial average
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        
        # Wilder's smoothing
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_12h = rsi(close, 14)
    rsi_long = rsi_12h > 50
    rsi_short = rsi_12h < 50
    
    # Volume filter (12h timeframe)
    vol_ma_20 = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20+1:i+1])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_dir_12h[i]) or np.isnan(rsi_long[i]) or 
            np.isnan(rsi_short[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, volume spike
            if kama_dir_12h[i] == 1 and rsi_long[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, volume spike
            elif kama_dir_12h[i] == -1 and rsi_short[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down OR RSI < 50
            if kama_dir_12h[i] == -1 or not rsi_long[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up OR RSI > 50
            if kama_dir_12h[i] == 1 or not rsi_short[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals