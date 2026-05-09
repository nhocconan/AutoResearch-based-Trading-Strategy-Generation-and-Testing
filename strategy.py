#!/usr/bin/env python3
# 4h_Choppiness_Donchian_Breakout_1dTrend
# Strategy: Donchian breakout with 1d EMA trend filter and choppiness regime filter
# Long when price breaks above Donchian(20) high, EMA50 > EMA200 on 1d, and CHOP(14) > 61.8 (range)
# Short when price breaks below Donchian(20) low, EMA50 < EMA200 on 1d, and CHOP(14) > 61.8
# Exit when price crosses 10-period EMA in opposite direction or CHOP < 38.2 (trend)
# Uses choppiness to avoid whipsaws in strong trends and capture breakouts in ranging markets
# Designed for 4h timeframe with selective entries to minimize trade frequency and maximize edge

name = "4h_Choppiness_Donchian_Breakout_1dTrend"
timeframe = "4h"
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
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Donchian channels (20-period)
    def donchian_channel(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate EMA(10) for exit signal
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Choppiness Index (14-period)
    def choppiness_index(high, low, close, period):
        chop = np.full_like(close, np.nan)
        atr = np.full_like(close, np.nan)
        
        # Calculate True Range
        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth TR with Wilder's method (equivalent to EMA with alpha=1/period)
        atr_smoothed = np.full_like(close, np.nan)
        if len(tr) >= period:
            atr_smoothed[period-1] = np.nanmean(tr[0:period])
            for i in range(period, len(tr)):
                atr_smoothed[i] = (atr_smoothed[i-1] * (period-1) + tr[i]) / period
        
        # Calculate Choppiness Index
        for i in range(period-1, len(close)):
            if not np.isnan(atr_smoothed[i]):
                max_high = np.max(high[i-period+1:i+1])
                min_low = np.min(low[i-period+1:i+1])
                if max_high > min_low and atr_smoothed[i] > 0:
                    chop[i] = 100 * np.log10((atr_smoothed[i] * period) / (max_high - min_low)) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 200)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(ema10[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Donchian breakout up, 1d uptrend, ranging market (high chop)
            if (close[i] > donchian_upper[i] and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i] and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakout down, 1d downtrend, ranging market (high chop)
            elif (close[i] < donchian_lower[i] and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i] and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA10 or chop indicates trend (low chop)
            if close[i] < ema10[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA10 or chop indicates trend (low chop)
            if close[i] > ema10[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals