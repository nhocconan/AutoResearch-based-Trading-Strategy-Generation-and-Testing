#!/usr/bin/env python3
# 4h_KAMA_Trend_Filter_Donchian_Breakout_With_Volume
# Hypothesis: Use KAMA on 1d to determine long/short regime, then enter on 4h Donchian breakout
# with volume confirmation. Exit on opposite Donchian breakout or when KAMA trend flips.
# Works in bull (breakout long) and bear (breakout short) by using KAMA to avoid counter-trend.
# KAMA adapts to choppy markets, reducing false breakouts. Targets 20-30 trades/year.

name = "4h_KAMA_Trend_Filter_Donchian_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(daily_close, prepend=daily_close[0]))
    volatility = np.sum(np.abs(np.diff(daily_close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    # Initialize KAMA
    kama = np.full_like(daily_close, np.nan, dtype=np.float64)
    kama[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
    
    # Donchian channels on daily (20-period)
    def rolling_max(arr, window):
        return np.convolve(arr, np.ones(window), 'valid')[:len(arr)-window+1] if len(arr) >= window else np.full_like(arr, np.nan)
    def rolling_min(arr, window):
        return np.convolve(arr, np.ones(window), 'valid')[:len(arr)-window+1] if len(arr) >= window else np.full_like(arr, np.nan)
    
    # Pad arrays for alignment
    if len(daily_high) >= 20:
        donchian_high = np.concatenate([np.full(19, np.nan), 
                                       np.array([np.max(daily_high[i-19:i+1]) for i in range(19, len(daily_high))])])
        donchian_low = np.concatenate([np.full(19, np.nan),
                                      np.array([np.min(daily_low[i-19:i+1]) for i in range(19, len(daily_low))])])
    else:
        donchian_high = np.full_like(daily_high, np.nan)
        donchian_low = np.full_like(daily_low, np.nan)
    
    # Align daily indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: 50-period moving average on 4h
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high with volume > 1.5x average and close > KAMA (uptrend)
            if close[i] > upper and volume[i] > 1.5 * vol_ma_val and close[i] > kama_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume > 1.5x average and close < KAMA (downtrend)
            elif close[i] < lower and volume[i] > 1.5 * vol_ma_val and close[i] < kama_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR close < KAMA (trend change)
            if close[i] < lower or close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR close > KAMA (trend change)
            if close[i] > upper or close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals