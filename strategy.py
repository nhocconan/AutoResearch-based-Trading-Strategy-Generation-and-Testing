#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout
# Uses 1d Donchian(20) breakout for structure, 4h Choppiness Index(14) to filter choppy markets
# Only trade when CHOP > 61.8 (range) for mean reversion or CHOP < 38.2 (trend) for trend following
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Target 80-160 total trades over 4 years (20-40/year) to avoid fee drag
# Works in bull/bear: adapts to regime, avoids whipsaw in strong trends, captures reversals in ranges

name = "4h_Choppiness_Regime_Donchian20_1d_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Calculate 4h Choppiness Index(14)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        atr_list = []
        for i in range(len(high_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(
                    high_arr[i] - low_arr[i],
                    abs(high_arr[i] - close_arr[i-1]),
                    abs(low_arr[i] - close_arr[i-1])
                )
            atr_list.append(tr)
        
        atr_arr = np.array(atr_list)
        tr_sum = pd.Series(atr_arr).rolling(window=window, min_periods=window).sum().values
        max_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        min_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        range_max_min = max_high - min_low
        
        chop = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if tr_sum[i] > 0 and range_max_min[i] > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / range_max_min[i]) / np.log10(window)
            else:
                chop[i] = 50.0  # neutral when undefined
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Align HTF indicators to 4h timeframe (primary)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or np.isnan(chop[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion in range: CHOP > 61.8 (range) + price at Donchian bounds
            if chop[i] > 61.8:
                if close[i] <= low_20_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= high_20_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            # Trend following in trend: CHOP < 38.2 (trend) + breakout
            elif chop[i] < 38.2:
                if close[i] > high_20_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midline or CHOP signals regime change
            midline = (low_20_aligned[i] + high_20_aligned[i]) / 2
            if close[i] <= midline or chop[i] > 61.8:  # exited range or reversed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midline or CHOP signals regime change
            midline = (low_20_aligned[i] + high_20_aligned[i]) / 2
            if close[i] >= midline or chop[i] > 61.8:  # exited range or reversed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals