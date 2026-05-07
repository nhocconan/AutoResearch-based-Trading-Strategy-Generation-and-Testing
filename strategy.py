#!/usr/bin/env python3
# 4h_Donchian_20_Trend_Volume_200MA
# Hypothesis: 4h Donchian(20) breakout with 200-period moving average trend filter and volume confirmation.
# Works in bull markets via breakout momentum and in bear markets via mean-reversion when price rejects the band.
# Volume confirms breakout strength, reducing false signals. Targets 20-50 trades per year on 4h timeframe.

name = "4h_Donchian_20_Trend_Volume_200MA"
timeframe = "4h"
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
    
    # 200-period MA for trend filter (using 4h data)
    ma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Donchian channels (20-period) - calculated on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any critical value is NaN
        if np.isnan(ma_200[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian band, above 200 MA, volume spike
            if close[i] > high_roll[i] and close[i] > ma_200[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band, below 200 MA, volume spike
            elif close[i] < low_roll[i] and close[i] < ma_200[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below lower Donchian band or below 200 MA
            if close[i] < low_roll[i] or close[i] < ma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above upper Donchian band or above 200 MA
            if close[i] > high_roll[i] or close[i] > ma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals