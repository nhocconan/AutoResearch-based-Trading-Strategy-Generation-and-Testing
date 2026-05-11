#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA calculation
    def kama(close, length=10, fast=2, slow=30):
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close, length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / np.where(volatility[length-1:] == 0, 1, volatility[length-1:])
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama_arr = np.full_like(close, np.nan)
        kama_arr[length] = close[length]
        for i in range(length+1, len(close)):
            kama_arr[i] = kama_arr[i-1] + sc[i] * (close[i] - kama_arr[i-1])
        return kama_arr
    
    # Calculate KAMA
    kama_arr = kama(close, length=10, fast=2, slow=30)
    kama_arr = np.where(np.isnan(kama_arr), close, kama_arr)  # Fill initial values
    
    # KAMA direction: 1 if close > KAMA, -1 if close < KAMA
    kama_dir = np.where(close > kama_arr, 1, -1)
    
    # RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        
        # Wilder smoothing
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi_arr = 100 - (100 / (1 + rs))
        return rsi_arr
    
    rsi_arr = rsi(close, length=14)
    
    # Choppiness Index calculation
    def choppy(high, low, close, length=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Sum of true ranges
        atr_sum = np.zeros_like(close)
        for i in range(length, len(close)):
            atr_sum[i] = np.sum(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        # Choppy calculation
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if atr_sum[i] > 0:
                chop[i] = 100 * np.log10(highest_high[i] - lowest_low[i]) / np.log10(length) / np.log10(atr_sum[i])
            else:
                chop[i] = 50
        return chop
    
    chop = choppy(high, low, close, length=14)
    
    # 1W EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after indicators are ready
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama_dir[i]) or np.isnan(rsi_arr[i]) or 
            np.isnan(chop[i]) or np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI oversold, not in choppy market
            if kama_dir[i] == 1 and rsi_arr[i] < 30 and chop[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, not in choppy market
            elif kama_dir[i] == -1 and rsi_arr[i] > 70 and chop[i] < 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or RSI overbought
            if kama_dir[i] == -1 or rsi_arr[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or RSI oversold
            if kama_dir[i] == 1 or rsi_arr[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals