#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: Trade Donchian channel breakouts on 4h timeframe in the direction of the 1d trend, with volume confirmation. This strategy aims to capture strong trending moves while avoiding false breakouts in sideways markets. Works in both bull and bear markets by aligning with higher timeframe trend. Target: 20-50 trades per year.

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend direction: 1 if close > EMA50, -1 if close < EMA50
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 4h Donchian channel (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend turns bearish
            if close[i] < lowest_low[i] or trend_1d_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend turns bullish
            if close[i] > highest_high[i] or trend_1d_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper band with volume confirmation and bullish 1d trend
            if (high[i] > highest_high[i] and vol_confirm[i] and trend_1d_aligned[i] == 1):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band with volume confirmation and bearish 1d trend
            elif (low[i] < lowest_low[i] and vol_confirm[i] and trend_1d_aligned[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals