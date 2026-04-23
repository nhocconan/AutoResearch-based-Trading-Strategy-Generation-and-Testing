#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1-day trend filter and volume confirmation.
Long when price breaks above Camarilla R1, 1-day EMA34 > EMA200, and volume > 1.3x average.
Short when price breaks below Camarilla S1, 1-day EMA34 < EMA200, and volume > 1.3x average.
Exit when price returns to Camarilla P (pivot) or trend reverses.
Camarilla levels are effective intraday support/resistance; EMA crossover filters trend direction.
Designed for low trade frequency (~25-40/year) to capture trend continuation with confirmation.
Works in both bull and bear markets by requiring EMA trend alignment for breakouts.
"""

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
    
    # Load 1-day data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1-day EMA34 and EMA200
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 4-hour data for Camarilla calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous day's Camarilla levels using prior day's OHLC
    # We need to shift by 1 to use previous day's data for today's levels
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]  # First value remains
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_val * 1.1 / 12)
    S1 = pivot - (range_val * 1.1 / 12)
    
    # Align HTF indicators to lower timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_aligned[i]
        ema200_val = ema200_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        pivot_val = pivot_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Price breaks above R1, EMA34 > EMA200 (uptrend), volume confirmation
            if (close_val > R1_val and
                ema34_val > ema200_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, EMA34 < EMA200 (downtrend), volume confirmation
            elif (close_val < S1_val and
                  ema34_val < ema200_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to pivot OR trend reverses (EMA34 < EMA200)
                if close_val <= pivot_val or ema34_val < ema200_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to pivot OR trend reverses (EMA34 > EMA200)
                if close_val >= pivot_val or ema34_val > ema200_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_1dEMA34_200_Volume"
timeframe = "4h"
leverage = 1.0