#!/usr/bin/env python3
# 12h_1dDonchian_Breakout_Trend_Filter_Volume
# Hypothesis: Donchian(20) breakouts on 12h timeframe with 1d trend filter (EMA50) and volume confirmation
# work in both bull and bear markets by capturing breakouts with trend alignment while avoiding chop.
# Low trade frequency (target: 15-30/year) minimizes fee drag. Uses 1d HTF for trend and Donchian channels.

name = "12h_1dDonchian_Breakout_Trend_Filter_Volume"
timeframe = "12h"
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
    
    # Get daily data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_donchian = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume average (20-period) for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    upper_donchian_12h = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_12h = align_htf_to_ltf(prices, df_1d, lower_donchian)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate volume spike on 12h timeframe
    vol_ma_20_12h_calc = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20_12h_calc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_donchian_12h[i]) or np.isnan(lower_donchian_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume
            if close[i] > upper_donchian_12h[i] and close[i] > ema_50_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume
            elif close[i] < lower_donchian_12h[i] and close[i] < ema_50_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below EMA50 (trend change)
            if close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above EMA50 (trend change)
            if close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals