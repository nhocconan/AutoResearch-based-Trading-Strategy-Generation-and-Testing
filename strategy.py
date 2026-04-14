#!/usr/bin/env python3
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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 60-period Exponential Moving Average (daily)
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema60 = ema(close_1d, 60)
    ema60_aligned = align_htf_to_ltf(prices, df_1d, ema60)
    
    # Calculate 14-period RSI (daily)
    def rsi(arr, period):
        if len(arr) < period + 1:
            return np.full_like(arr, np.nan)
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(arr, np.nan)
        avg_loss = np.full_like(arr, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi14 = rsi(close_1d, 14)
    rsi14_aligned = align_htf_to_ltf(prices, df_1d, rsi14)
    
    # Calculate 30-period volume moving average (daily)
    vol_ma_30 = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 30:
        for i in range(29, len(volume_1d)):
            vol_ma_30[i] = np.mean(volume_1d[i-29:i+1])
    vol_ma_30_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_30)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema60_aligned[i]) or 
            np.isnan(rsi14_aligned[i]) or 
            np.isnan(vol_ma_30_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 30-day average volume
        if vol_ma_30_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_30_aligned[i]
        
        if position == 0:
            # Long: Price above EMA60 + RSI > 50 + volume surge
            if (close[i] > ema60_aligned[i] and
                rsi14_aligned[i] > 50 and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price below EMA60 + RSI < 50 + volume surge
            elif (close[i] < ema60_aligned[i] and
                  rsi14_aligned[i] < 50 and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price below EMA60 OR RSI < 40
            if (close[i] < ema60_aligned[i] or 
                rsi14_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price above EMA60 OR RSI > 60
            if (close[i] > ema60_aligned[i] or 
                rsi14_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_EMA60_RSI14_Volume"
timeframe = "6h"
leverage = 1.0