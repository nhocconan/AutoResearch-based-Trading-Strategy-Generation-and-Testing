#!/usr/bin/env python3
# Hypothesis: 4h 200-period SMA trend filter with price closing above/below SMA for entry.
# Uses long-term trend from 200-period SMA on 4h chart to capture sustained moves in both bull and bear markets.
# Entry when price closes above SMA for long, below SMA for short, with volume confirmation to avoid false breakouts.
# Exit when price closes back below/above SMA respectively. Designed for low trade frequency (<25/year) to minimize fee drag.
# Trend following with SMA has shown robustness in trending markets while avoiding whipsaws in ranging conditions.

name = "4h_SMA200_Trend_Following"
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
    
    # Calculate 4h 200-period SMA for trend filter
    close_series = pd.Series(close)
    sma200 = close_series.rolling(window=200, min_periods=200).mean().values
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after sufficient data for SMA200
        if np.isnan(sma200[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above SMA200 with volume confirmation
            if close[i] > sma200[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below SMA200 with volume confirmation
            elif close[i] < sma200[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back below SMA200
            if close[i] < sma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back above SMA200
            if close[i] > sma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals