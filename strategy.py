#!/usr/bin/env python3
# 4H_Donchian_Breakout_Volume_Trend_12h
# Hypothesis: Donchian channel (20-period high/low) breakouts signal strong momentum.
# Entry requires breakout above upper band (long) or below lower band (short) with volume > 1.5x 20-period average.
# Trend filter uses 12h EMA50 to align with higher timeframe trend, reducing false signals in chop.
# Exit occurs when price reverts to the 20-period EMA, capturing mean reversion within the trend.
# Designed for low trade frequency (~20-40/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull/bear markets by following the higher timeframe trend.

name = "4H_Donchian_Breakout_Volume_Trend_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel: 20-period high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # 12h trend filter: EMA 50
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Exit EMA: 20-period EMA for mean reversion exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above Donchian upper band + volume confirmation + 12h uptrend
            if close[i] > high_20[i] and volume[i] > vol_threshold[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian lower band + volume confirmation + 12h downtrend
            elif close[i] < low_20[i] and volume[i] > vol_threshold[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below 20-period EMA (mean reversion)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above 20-period EMA (mean reversion)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals