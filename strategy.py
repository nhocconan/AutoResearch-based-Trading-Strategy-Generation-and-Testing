#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: Donchian(20) breakout on 1d with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high and weekly close > weekly open (bullish week).
Short when price breaks below 20-day low and weekly close < weekly open (bearish week).
Volume confirmation filters weak signals. Works in bull/bear by following higher timeframe trend.
Target: 10-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly bullish/bearish: close > open = bullish, close < open = bearish
    weekly_bullish = df_1w['close'] > df_1w['open']
    weekly_bearish = df_1w['close'] < df_1w['open']
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.values.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.values.astype(float))
    
    # Donchian(20) channels on 1d
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low or weekly trend turns bearish
            if close[i] < low_20[i] or weekly_bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or weekly trend turns bullish
            if close[i] > high_20[i] or weekly_bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day high with volume and bullish weekly trend
            if (close[i] > high_20[i] and vol_confirm and 
                weekly_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-day low with volume and bearish weekly trend
            elif (close[i] < low_20[i] and vol_confirm and 
                  weekly_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals