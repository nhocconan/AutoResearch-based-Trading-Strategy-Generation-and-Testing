#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: Use 12h Camarilla pivot levels for directional bias on 4h chart, enter on breakout of H4/L4 with volume confirmation and choppiness regime filter to avoid whipsaws. Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag. Works in bull via breakouts, in bear via mean-reversion from extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla levels and choppiness
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla calculation
    prev_high = df_12h['high'].iloc[-2] if len(df_12h) >= 2 else df_12h['high'].iloc[-1]
    prev_low = df_12h['low'].iloc[-2] if len(df_12h) >= 2 else df_12h['low'].iloc[-1]
    prev_close = df_12h['close'].iloc[-2] if len(df_12h) >= 2 else df_12h['close'].iloc[-1]
    
    # Calculate 12h Camarilla levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    daily_h4 = prev_close + 1.1 * range_val * 1.1 / 2
    daily_l4 = prev_close - 1.1 * range_val * 1.1 / 2
    
    # Align 12h levels to 4h timeframe
    daily_h4_array = np.full(len(df_12h), daily_h4)
    daily_l4_array = np.full(len(df_12h), daily_l4)
    daily_h4_aligned = align_htf_to_ltf(prices, df_12h, daily_h4_array)
    daily_l4_aligned = align_htf_to_ltf(prices, df_12h, daily_l4_array)
    
    # 12h Choppiness Index for regime filter
    def calculate_choppiness(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = np.abs(high_arr[i] - close_arr[i-1])
            lc = np.abs(low_arr[i] - close_arr[i-1])
            tr[i] = max(hl, hc, lc)
        # Wilder's smoothing for ATR
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Sum of absolute true range over period
        atr_sum = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Choppiness formula
        chop = np.zeros_like(close_arr)
        for i in range(period, len(close_arr)):
            if atr_sum[i] > 0:
                max_h = np.max(high_arr[i-period+1:i+1])
                min_l = np.min(low_arr[i-period+1:i+1])
                chop[i] = 100 * np.log10(atr_sum[i] / (max_h - min_l)) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_choppiness(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume moving average (20-period) for confirmation
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(daily_h4_aligned[i]) or np.isnan(daily_l4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: chop > 50 indicates ranging market (good for mean reversion)
        # We use chop > 50 for both long and short to avoid strong trends
        regime_filter = chop_aligned[i] > 50
        
        # Breakout conditions with filters
        long_breakout = (close[i] > daily_h4_aligned[i]) and volume_confirm and regime_filter
        short_breakout = (close[i] < daily_l4_aligned[i]) and volume_confirm and regime_filter
        
        # Exit conditions: return to midpoint between H4 and L4
        midpoint = (daily_h4_aligned[i] + daily_l4_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals