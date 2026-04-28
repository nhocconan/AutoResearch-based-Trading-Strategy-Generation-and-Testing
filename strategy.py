#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian upper (20) with 1d uptrend (EMA34 > EMA89) and volume spike, short when breaks below lower with 1d downtrend and volume spike. Exit on opposite Donchian break with volume. Trend filter avoids counter-trend trades; volume confirms institutional participation. Designed for ~20-40 trades/year to minimize fee drag in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 89:
        return np.zeros(n)
    
    # Calculate daily 34 and 89 EMA for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_daily = pd.Series(close_daily).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align daily EMAs to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    ema89_daily_aligned = align_htf_to_ltf(prices, df_daily, ema89_daily)
    
    # Daily trend: bullish when EMA34 > EMA89
    daily_uptrend = ema34_daily_aligned > ema89_daily_aligned
    daily_downtrend = ema34_daily_aligned < ema89_daily_aligned
    
    # Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(ema89_daily_aligned[i]) or
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume spike
        long_entry = close[i] > high_max_20[i] and daily_uptrend[i] and volume_spike[i]
        short_entry = close[i] < low_min_20[i] and daily_downtrend[i] and volume_spike[i]
        
        # Exit on opposite Donchian break with volume spike
        long_exit = close[i] < low_min_20[i] and volume_spike[i]
        short_exit = close[i] > high_max_20[i] and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0