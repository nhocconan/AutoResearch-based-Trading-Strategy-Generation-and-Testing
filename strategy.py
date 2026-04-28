#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on 1d
    def calculate_kama(arr, period=10):
        # Efficiency Ratio (ER)
        change = np.abs(np.diff(arr, n=period))
        volatility = np.sum(np.abs(np.diff(arr)), axis=0) if len(arr) > 1 else 0
        # For vectorized calculation
        er = np.zeros_like(arr)
        for i in range(period, len(arr)):
            change_val = np.abs(arr[i] - arr[i-period])
            volatility_val = np.sum(np.abs(np.diff(arr[i-period:i+1])))
            if volatility_val != 0:
                er[i] = change_val / volatility_val
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
        # Initialize KAMA
        kama = np.full_like(arr, np.nan)
        kama[period] = arr[period]
        for i in range(period+1, len(arr)):
            kama[i] = kama[i-1] + sc[i] * (arr[i] - kama[i-1])
        return kama
    
    kama_1d = calculate_kama(close_1d, 10)
    
    # RSI on 1d
    def calculate_rsi(arr, period=14):
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Choppiness Index on 1d
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = tr
        # Smoothed ATR (simple moving average)
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop = np.full_like(close, np.nan)
        for i in range(period, len(close)):
            if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Align to 4h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_surge = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        
        # RSI conditions: not overbought/oversold
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Chop regime: chop > 50 indicates ranging market (good for mean reversion)
        chop_ranging = chop_1d_aligned[i] > 50
        
        # Entry conditions
        # Long: price above KAMA + RSI not overbought + chop ranging + volume surge
        long_entry = price_above_kama and rsi_not_overbought and chop_ranging and volume_surge[i]
        # Short: price below KAMA + RSI not oversold + chop ranging + volume surge
        short_entry = price_below_kama and rsi_not_oversold and chop_ranging and volume_surge[i]
        
        # Exit conditions: opposite conditions or chop trending (< 50)
        chop_trending = chop_1d_aligned[i] < 50
        long_exit = price_below_kama or rsi_1d_aligned[i] > 70 or chop_trending
        short_exit = price_above_kama or rsi_1d_aligned[i] < 30 or chop_trending
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA10_RSI14_Chop14_Volume"
timeframe = "4h"
leverage = 1.0