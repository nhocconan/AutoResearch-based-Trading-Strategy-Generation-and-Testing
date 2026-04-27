#!/usr/bin/env python3
# 101013
# 4h_Donchian20_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel in uptrend (price > EMA50) with volume spike.
# Short when price breaks below lower Donchian channel in downtrend (price < EMA50) with volume spike.
# Uses 12h EMA50 to filter trend direction - avoids counter-trend breakouts that fail in consolidation.
# Volume spike ensures breakout has institutional participation.
# Target: 20-50 trades/year to minimize fee drag while capturing strong trends.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    # Using rolling window on high/low for upper/lower bands
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian, above EMA50 (uptrend), volume spike
        if (close[i] > donchian_upper[i] and 
            close[i] > ema50_12h_aligned[i] and 
            volume_spike[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian, below EMA50 (downtrend), volume spike
        elif (close[i] < donchian_lower[i] and 
              close[i] < ema50_12h_aligned[i] and 
              volume_spike[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite Donchian level (mean reversion)
        elif position == 1 and close[i] < donchian_lower[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_upper[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0