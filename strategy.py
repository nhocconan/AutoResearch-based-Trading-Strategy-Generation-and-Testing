#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Enter long when price breaks above Donchian upper in 1d uptrend with volume spike.
# Enter short when price breaks below Donchian lower in 1d downtrend with volume spike.
# Uses 1d EMA(50) for trend filter and 30-period volume spike (1.8x EMA) for confirmation.
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = ema_50_1d[1:] > ema_50_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 30-period volume spike (1.8x EMA)
    vol_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    # Donchian(20) channels on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for Donchian and volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: breakout above Donchian upper in uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] >= donchian_upper[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: breakdown below Donchian lower in downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] <= donchian_lower[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reversal or breakdown below lower band
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                close[i] <= donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or breakout above upper band
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] >= donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals