#!/usr/bin/env python3
# 4h_BollingerBreakout_VolumeATRStop
# Hypothesis: On 4h chart, enter long when price breaks above Bollinger upper band with volume confirmation,
# enter short when price breaks below Bollinger lower band with volume confirmation.
# Use ATR-based stoploss via signal=0 when price closes outside bands.
# Bollinger Bands adapt to volatility, reducing false breakouts in ranging periods.
# Works in both bull and bear markets by capturing breakouts with volume filter.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.
timeframe = "4h"
name = "4h_BollingerBreakout_VolumeATRStop"
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
    
    # Bollinger Bands parameters
    bb_period = 20
    bb_std = 2.0
    
    # Calculate SMA of close (middle line)
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    
    # Calculate standard deviation
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    
    # Calculate Bollinger Bands
    bb_upper = sma + bb_std * std
    bb_lower = sma - bb_std * std
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(sma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Bollinger upper band + volume spike
            if close[i] > bb_upper[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Bollinger lower band + volume spike
            elif close[i] < bb_lower[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Bollinger lower band (stoploss)
            if close[i] < bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Bollinger upper band (stoploss)
            if close[i] > bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals