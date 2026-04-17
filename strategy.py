#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA 8 and SMA 21
    sma8_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    sma21_1w = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
    
    # Align weekly SMAs to 6h timeframe
    sma8_1w_aligned = align_htf_to_ltf(prices, df_1w, sma8_1w)
    sma21_1w_aligned = align_htf_to_ltf(prices, df_1w, sma21_1w)
    
    # Weekly trend: bullish if SMA8 > SMA21, bearish if SMA8 < SMA21
    weekly_bullish = sma8_1w_aligned > sma21_1w_aligned
    weekly_bearish = sma8_1w_aligned < sma21_1w_aligned
    
    # Get daily data for ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily True Range and ATR(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily range for dynamic thresholds
    daily_range = high_1d - low_1d
    range_ma10 = pd.Series(daily_range).rolling(window=10, min_periods=10).mean().values
    range_ma10_aligned = align_htf_to_ltf(prices, df_1d, range_ma10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need sufficient data for weekly and daily calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma8_1w_aligned[i]) or np.isnan(sma21_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(range_ma10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Dynamic threshold based on volatility
        vol_threshold = 0.5 * range_ma10_aligned[i]
        
        if position == 0:
            # Long entry: weekly bullish + price above weekly SMA8 + volatility filter
            if (weekly_bullish[i] and 
                close[i] > sma8_1w_aligned[i] + vol_threshold):
                signals[i] = 0.25
                position = 1
            # Short entry: weekly bearish + price below weekly SMA21 + volatility filter
            elif (weekly_bearish[i] and 
                  close[i] < sma21_1w_aligned[i] - vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly SMA21 or weekly trend turns bearish
            if (close[i] < sma21_1w_aligned[i] or 
                weekly_bearish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly SMA8 or weekly trend turns bullish
            if (close[i] > sma8_1w_aligned[i] or 
                weekly_bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6s_WeeklySMA_Trend_Follow"
timeframe = "6h"
leverage = 1.0