#!/usr/bin/env python3
# 4h_KeltnerChannel_Breakout_ATRStop_Volume
# Hypothesis: On 4h chart, enter long when price breaks above Keltner upper band with volume confirmation,
# enter short when price breaks below Keltner lower band with volume confirmation.
# Use ATR-based stoploss via signal=0 when price closes outside bands.
# Designed for low trade frequency (~20-40/year) to minimize fee drag and work in trending markets.
# Keltner channels adapt to volatility, reducing false breakouts in ranging periods.
# Works in both bull and bear markets by capturing breakouts with volume filter.
timeframe = "4h"
name = "4h_KeltnerChannel_Breakout_ATRStop_Volume"
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
    
    # Keltner Channel parameters
    kc_period = 20
    kc_multiplier = 2.0
    
    # Calculate EMA of close (middle line)
    ema = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Calculate ATR
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Calculate Keltner Bands
    kc_upper = ema + kc_multiplier * atr
    kc_lower = ema - kc_multiplier * atr
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(kc_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner upper band + volume spike
            if close[i] > kc_upper[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner lower band + volume spike
            elif close[i] < kc_lower[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Keltner lower band (stoploss)
            if close[i] < kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Keltner upper band (stoploss)
            if close[i] > kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals