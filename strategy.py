#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: Camarilla pivot levels from 1d + volume spike + choppiness regime filter on 12h timeframe.
# Long: Close > H3 AND volume > 1.5 * volume_ma(20) AND CHOP(14) > 61.8 (range regime)
# Short: Close < L3 AND volume > 1.5 * volume_ma(20) AND CHOP(14) > 61.8 (range regime)
# Exit: Opposite pivot touch or regime change to trending (CHOP < 38.2)
# Uses 12h primary timeframe with 1d HTF for Camarilla levels and chop filter.
# Target: 50-150 total trades over 4 years to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
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
    
    # Calculate volume moving average for spike detection
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 12h data
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros(len(close_arr))
        tr = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]),
                       abs(low_arr[i] - close_arr[i-1]))
        # Smooth TR with Wilder's smoothing (EMA with alpha=1/period)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Calculate chopiness
        chop = np.full(len(close_arr), np.nan)
        for i in range(period, len(close_arr)):
            atr_sum = np.nansum(atr[i-period+1:i+1])
            hh = np.nanmax(high_arr[i-period+1:i+1])
            ll = np.nanmin(low_arr[i-period+1:i+1])
            if hh != ll and atr_sum > 0:
                chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    camarilla_high = np.zeros(len(close_1d))
    camarilla_low = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_high[i] = np.nan
            camarilla_low[i] = np.nan
        else:
            # Camarilla levels based on previous day
            range_ = high_1d[i-1] - low_1d[i-1]
            camarilla_high[i] = close_1d[i-1] + range_ * 1.1 / 4  # H3
            camarilla_low[i] = close_1d[i-1] - range_ * 1.1 / 4   # L3
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i])):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        in_range = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price touches L3 (opposite pivot) OR regime changes to trending
            if close[i] < camarilla_low_aligned[i] or not in_range:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H3 (opposite pivot) OR regime changes to trending
            if close[i] > camarilla_high_aligned[i] or not in_range:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Close > H3 AND volume spike AND ranging regime
            if close[i] > camarilla_high_aligned[i] and volume_spike and in_range:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < L3 AND volume spike AND ranging regime
            elif close[i] < camarilla_low_aligned[i] and volume_spike and in_range:
                position = -1
                signals[i] = -0.25
    
    return signals