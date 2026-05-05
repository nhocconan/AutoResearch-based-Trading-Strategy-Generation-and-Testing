#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian(20) breakout captures strong momentum moves in both bull/bear markets
# 1w EMA50 ensures we only trade with the primary weekly trend (avoid counter-trend)
# Volume confirmation (>1.5x 20-period MA) adds conviction to breakouts
# Exit: Donchian(20) opposite breakout or weekly trend reversal
# Timeframe: 12h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) >= 20 and len(low) >= 20:
        # Upper channel: highest high over past 20 periods
        high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Lower channel: lowest low over past 20 periods
        low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        high_rolling_max = np.full(n, np.nan)
        low_rolling_min = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: close > upper Donchian AND price > 1w EMA50 AND volume spike
            if (close[i] > high_rolling_max[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: close < lower Donchian AND price < 1w EMA50 AND volume spike
            elif (close[i] < low_rolling_min[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close < lower Donchian OR price < 1w EMA50 (trend reversal)
            if close[i] < low_rolling_min[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close > upper Donchian OR price > 1w EMA50 (trend reversal)
            if close[i] > high_rolling_max[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals