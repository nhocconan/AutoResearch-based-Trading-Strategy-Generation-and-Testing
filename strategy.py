#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Filter_Volume
# Hypothesis: Use 1d KAMA to determine trend direction, enter long when price crosses above KAMA with RSI > 50 and volume confirmation, short when price crosses below KAMA with RSI < 50 and volume confirmation. Exit when price re-crosses KAMA. This strategy aims to capture trends with momentum and volume confirmation, works in both bull and bear markets by following the trend.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_KAMA_Trend_RSI_Filter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on 1d
    def kama(close, length=10, fast=2, slow=30):
        # Calculate efficiency ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # Initialize KAMA
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_values = kama(close, 10, 2, 30)
    
    # Calculate RSI on 1d
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        # Pad the beginning
        rsi_out = np.concatenate([np.full(length, np.nan), rsi_out[length:]])
        return rsi_out
    
    rsi_values = rsi(close, 14)
    
    # Align 1w close for trend filter
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_values[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(close_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get aligned 1w close for trend filter
        close_1w_current = close_1w_aligned[i]
        
        # Trend filter: only trade in direction of 1w trend
        # For simplicity, use price vs 1w close (could be improved with MA)
        trend_up = close[i] > close_1w_current
        trend_down = close[i] < close_1w_current
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Price crosses above KAMA with RSI > 50, volume confirmation, and uptrend
            if close[i] > kama_values[i] and rsi_values[i] > 50 and vol_confirm and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA with RSI < 50, volume confirmation, and downtrend
            elif close[i] < kama_values[i] and rsi_values[i] < 50 and vol_confirm and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals