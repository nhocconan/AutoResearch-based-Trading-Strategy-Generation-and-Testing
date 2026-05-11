#!/usr/bin/env python3
# 4h_1d_Donchian_Breakout_Volume_Filter
# Hypothesis: Donchian(20) breakouts with volume confirmation and trend filter from 1d EMA50.
# Works in bull via breakouts above upper band in uptrend, and in bear via breakdowns below lower band in downtrend.
# Volume filter reduces false breakouts. Trend filter ensures alignment with higher timeframe bias.
# Designed for low trade frequency (20-40/year) to minimize fee drag.

name = "4h_1d_Donchian_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope_1d = np.diff(ema_50_1d, prepend=ema_50_1d[0])  # slope = today - yesterday
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d EMA50 slope
        bullish_trend = ema_slope_1d_aligned[i] > 0
        bearish_trend = ema_slope_1d_aligned[i] < 0
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above upper Donchian band in bullish trend with volume surge
            if close[i] > highest_high[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower Donchian band in bearish trend with volume surge
            elif close[i] < lowest_low[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit: close below midpoint of Donchian channel
                midpoint = (highest_high[i] + lowest_low[i]) / 2.0
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: close above midpoint of Donchian channel
                midpoint = (highest_high[i] + lowest_low[i]) / 2.0
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals