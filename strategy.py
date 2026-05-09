#!/usr/bin/env python3
name = "12h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend direction
    def kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / volatility[period-1:]
        er = np.where(volatility == 0, 0, er)
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_vals = np.zeros_like(price)
        kama_vals[period] = price[period]
        for i in range(period+1, len(price)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (price[i-1] - kama_vals[i-1])
        return kama_vals
    
    # Calculate RSI
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[0], tr])
        
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama_vals = kama(close, period=10, fast=2, slow=30)
    rsi_vals = rsi(close, period=14)
    chop_vals = choppiness_index(high, low, close, period=14)
    
    # Get daily RSI and chop for filter
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    rsi_1d = rsi(close_1d, period=14)
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, period=14)
    
    # Align daily indicators to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(20, len(volume)):
        volume_avg[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        # Skip if KAMA data not ready
        if np.isnan(kama_vals[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if daily RSI or chop not ready
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising + RSI > 50 + chop < 61.8 (trending) + volume confirmation
            if (kama_vals[i] > kama_vals[i-1] and 
                rsi_1d_aligned[i] > 50 and 
                chop_1d_aligned[i] < 61.8 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling + RSI < 50 + chop < 61.8 (trending) + volume confirmation
            elif (kama_vals[i] < kama_vals[i-1] and 
                  rsi_1d_aligned[i] < 50 and 
                  chop_1d_aligned[i] < 61.8 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 45 OR chop > 61.8 (ranging)
            if (kama_vals[i] < kama_vals[i-1] or 
                rsi_1d_aligned[i] < 45 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 55 OR chop > 61.8 (ranging)
            if (kama_vals[i] > kama_vals[i-1] or 
                rsi_1d_aligned[i] > 55 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals