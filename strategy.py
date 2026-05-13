#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1w EMA34 AND volume > 1.5x 20-period average volume.
# Short when price breaks below Donchian(20) low AND price < 1w EMA34 AND volume > 1.5x 20-period average volume.
# Exit when price reverts to Donchian(20) midpoint OR trend filter reverses.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~7-25/year) by requiring confluence of breakout, HTF trend, and volume spike.
# Effective in both bull and bear markets by capturing strong directional moves with trend and volatility filters.

name = "1d_Donchian20_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Donchian
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(highest_high[i]) or \
           np.isnan(lowest_low[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian(20) high AND price > 1w EMA34 AND volume confirmation
            if close[i] > highest_high[i] and close[i] > ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian(20) low AND price < 1w EMA34 AND volume confirmation
            elif close[i] < lowest_low[i] and close[i] < ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reverts to Donchian(20) midpoint OR trend filter reverses (price < 1w EMA34)
            if close[i] <= donchian_mid[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reverts to Donchian(20) midpoint OR trend filter reverses (price > 1w EMA34)
            if close[i] >= donchian_mid[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals