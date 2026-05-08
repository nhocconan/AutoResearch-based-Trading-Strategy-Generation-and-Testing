#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and KAMA trend filter.
# Weekly Donchian channels capture long-term breakouts; volume confirms institutional participation.
# KAMA adapts to market conditions, reducing whipsaws in sideways markets.
# Designed for low trade frequency (7-25/year) to minimize fee drift while capturing major trends.
# Works in both bull and bear markets by filtering breakouts with trend alignment.

name = "1d_WeeklyDonchian_Breakout_Volume_KAMA"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper_1w = np.full_like(high_1w, np.nan)
    lower_1w = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        upper_1w[i] = np.max(high_1w[i-20:i])
        lower_1w[i] = np.min(low_1w[i-20:i])
    
    # Align Donchian channels to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # KAMA trend filter on daily timeframe
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series.diff(10))
    volatility = close_series.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: daily volume > 1.5x 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly upper Donchian with volume confirmation and above KAMA
            if close[i] > upper_aligned[i] and vol_confirm[i] and close[i] > kama[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly lower Donchian with volume confirmation and below KAMA
            elif close[i] < lower_aligned[i] and vol_confirm[i] and close[i] < kama[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly lower Donchian or below KAMA
            if close[i] < lower_aligned[i] or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly upper Donchian or above KAMA
            if close[i] > upper_aligned[i] or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals