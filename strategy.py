#!/usr/bin/env python3
# 4h_12h_Donchian_Breakout_Volume_Trend
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter.
# Enters long on breakout above 20-period high with volume > 1.5x average and price > 12h EMA50.
# Enters short on breakdown below 20-period low with volume > 1.5x average and price < 12h EMA50.
# Exits when price crosses the 10-period moving average in the opposite direction.
# Uses 12h EMA for trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-30/year) to minimize fee drag.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "4h_12h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(ema_10[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h EMA50
        # Need aligned 12h close price for trend comparison
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        uptrend = close_12h_aligned[i] > ema_50_aligned[i]
        downtrend = close_12h_aligned[i] < ema_50_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above 20-period high in uptrend with volume
            if close[i] > high_20[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below 20-period low in downtrend with volume
            elif close[i] < low_20[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price crosses below 10-period EMA
                if close[i] < ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above 10-period EMA
                if close[i] > ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals