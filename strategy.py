#!/usr/bin/env python3
"""
6h_1d_Percentile_Band_MeanReversion
Hypothesis: On 6h timeframe, mean-revert from extreme deviations of price from 1-day median price.
Uses 1-day rolling median and 84th/16th percentiles (equivalent to ±1 sigma in normal dist) as bands.
Enter long when price touches lower band with bullish divergence on RSI(6), short when price touches upper band with bearish RSI divergence.
Works in both bull and bear markets by fading extremes while respecting the 1-day trend via price vs 200-period EMA filter.
Designed for low trade frequency (~20-40/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Percentile_Band_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY INDICATORS: Median and percentile bands ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-period rolling median of daily close
    def rolling_median(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.median(arr[i - window + 1:i + 1])
        return result
    
    median_20 = rolling_median(close_1d, 20)
    
    # Calculate 84th and 16th percentiles (~1 sigma) over 20 days
    def rolling_percentile(arr, window, percentile):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.percentile(arr[i - window + 1:i + 1], percentile)
        return result
    
    upper_band = rolling_percentile(close_1d, 20, 84)
    lower_band = rolling_percentile(close_1d, 20, 16)
    
    # Align to 6h timeframe
    median_aligned = align_htf_to_ltf(prices, df_1d, median_20)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # === 6H INDICATORS: RSI(6) for divergence ===
    def rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(arr)
        avg_loss = np.zeros_like(arr)
        
        if len(arr) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            for i in range(period, len(arr)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_6 = rsi(close, 6)
    
    # === 1D EMA200 FOR TREND FILTER ===
    ema_200 = np.zeros_like(close_1d)
    if len(close_1d) >= 200:
        ema_200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200[i] = (close_1d[i] * 2 + ema_200[i-1] * 198) / 200
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if indicators not available
        if (np.isnan(median_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(rsi_6[i]) or 
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals with trend filter
        near_lower = close[i] <= lower_aligned[i] * 1.001  # slight buffer
        near_upper = close[i] >= upper_aligned[i] * 0.999
        
        # RSI divergence: bullish when RSI rising from oversold, bearish when falling from overbought
        rsi_rising = rsi_6[i] > rsi_6[i-1]
        rsi_falling = rsi_6[i] < rsi_6[i-1]
        
        # Trend filter: only long above EMA200, only short below EMA200
        trend_filter_long = close[i] > ema_200_aligned[i]
        trend_filter_short = close[i] < ema_200_aligned[i]
        
        long_signal = near_lower and rsi_rising and trend_filter_long
        short_signal = near_upper and rsi_falling and trend_filter_short
        
        # Exit when price returns to median or RSI reaches opposite extreme
        exit_long = close[i] >= median_aligned[i] or rsi_6[i] >= 60
        exit_short = close[i] <= median_aligned[i] or rsi_6[i] <= 40
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals