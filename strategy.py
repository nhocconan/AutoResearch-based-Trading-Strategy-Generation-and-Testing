#!/usr/bin/env python3
# 4h_PocFibo_Breakout_TrendVolume
# Hypothesis: 4h breakout at point of control (POC) or 61.8% Fibonacci level from prior swing, filtered by 1d EMA trend and volume spike confirmation. Uses volume profile POC and Fibonacci retracement of recent swing to identify high-probability support/resistance levels. Works in both bull and bear by following 1d trend direction, with volume confirming breakout strength. Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_PocFibo_Breakout_TrendVolume"
timeframe = "4h"
leverage = 1.0

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

    # Get 4h data for price action and swing detection
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate Volume Profile POC (Point of Control) for 4h - price level with highest volume in lookback
    def calculate_poc(high_arr, low_arr, close_arr, vol_arr, lookback=20):
        # Create price bins and accumulate volume
        price_min = np.min(low_arr[-lookback:]) if len(low_arr) >= lookback else np.min(low_arr)
        price_max = np.max(high_arr[-lookback:]) if len(high_arr) >= lookback else np.max(high_arr)
        if price_max <= price_min:
            price_max = price_min + 0.01
        
        # Create 50 price bins
        bins = np.linspace(price_min, price_max, 51)
        volume_profile = np.zeros(50)
        
        # Accumulate volume for each price level in lookback period
        start_idx = max(0, len(close_arr) - lookback)
        for i in range(start_idx, len(close_arr)):
            # Typical price for volume distribution
            typical_price = (high_arr[i] + low_arr[i] + close_arr[i]) / 3
            # Find bin index
            bin_idx = np.searchsorted(bins, typical_price) - 1
            bin_idx = max(0, min(bin_idx, 49))  # Clamp to valid range
            volume_profile[bin_idx] += vol_arr[i]
        
        # Find POC (price level with maximum volume)
        if np.max(volume_profile) > 0:
            poc_bin = np.argmax(volume_profile)
            poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
            return poc_price
        else:
            return close_arr[-1]  # fallback to close

    # Calculate POC for rolling window
    poc_values = np.full(len(close_4h), np.nan)
    lookback = 20
    for i in range(lookback, len(close_4h)):
        poc_values[i] = calculate_poc(high_4h[:i+1], low_4h[:i+1], close_4h[:i+1], volume_4h[:i+1], lookback)

    # Calculate Fibonacci retracement levels from recent swing
    def calculate_fib_levels(high_arr, low_arr, lookback=30):
        if len(high_arr) < lookback:
            return np.full(len(high_arr), np.nan), np.full(len(high_arr), np.nan)
        
        # Find swing high and low in lookback period
        start_idx = len(high_arr) - lookback
        swing_high_idx = np.argmax(high_arr[start_idx:]) + start_idx
        swing_low_idx = np.argmin(low_arr[start_idx:]) + start_idx
        
        swing_high = high_arr[swing_high_idx]
        swing_low = low_arr[swing_low_idx]
        diff = swing_high - swing_low
        
        if diff <= 0:
            return np.full(len(high_arr), np.nan), np.full(len(high_arr), np.nan)
        
        # 61.8% Fibonacci level (most important for retracements)
        fib_618 = swing_high - 0.618 * diff
        # Also calculate 38.2% for additional context
        fib_382 = swing_high - 0.382 * diff
        
        # Arrays to store levels
        fib_618_arr = np.full(len(high_arr), np.nan)
        fib_382_arr = np.full(len(high_arr), np.nan)
        
        # Fill from the point where swing is identified
        fill_start = max(swing_high_idx, swing_low_idx)
        fib_618_arr[fill_start:] = fib_618
        fib_382_arr[fill_start:] = fib_382
        
        return fib_618_arr, fib_382_arr

    fib_618_4h, fib_382_4h = calculate_fib_levels(high_4h, low_4h, lookback=30)

    # Calculate ATR for volatility filtering
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Align all indicators to 4h timeframe
    poc_aligned = align_htf_to_ltf(prices, df_4h, poc_values)
    fib_618_aligned = align_htf_to_ltf(prices, df_4h, fib_618_4h)
    fib_382_aligned = align_htf_to_ltf(prices, df_4h, fib_382_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)

    # Calculate 4h volume SMA for volume confirmation
    volume_series = pd.Series(volume_4h)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_sma_aligned = align_htf_to_ltf(prices, df_4h, volume_sma20)
    volume_spike_threshold = volume_sma_aligned * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after enough data for calculations
        # Skip if any required data is NaN
        if (np.isnan(poc_aligned[i]) or np.isnan(fib_618_aligned[i]) or 
            np.isnan(fib_382_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(volume_sma_aligned[i]) or np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume confirmation - current 4h volume
        current_4h_idx = i // 16  # Approximate 4h bar index
        if current_4h_idx >= len(volume):
            current_4h_idx = len(volume) - 1
        vol_ok = volume[i] > volume_sma_aligned[i]  # Volume above average

        if position == 0:
            # LONG: Price above POC and above 61.8% Fib level in 1d uptrend with volume confirmation
            if (close[i] > poc_aligned[i] and 
                close[i] > fib_618_aligned[i] and 
                close[i] > ema20_1d_aligned[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below POC and below 61.8% Fib level in 1d downtrend with volume confirmation
            elif (close[i] < poc_aligned[i] and 
                  close[i] < fib_618_aligned[i] and 
                  close[i] < ema20_1d_aligned[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below POC or below 38.2% Fib level (weaker support)
            if close[i] < poc_aligned[i] or close[i] < fib_382_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above POC or above 38.2% Fib level (weaker resistance)
            if close[i] > poc_aligned[i] or close[i] > fib_382_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals