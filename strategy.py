#!/usr/bin/env python3
"""
#100781 - 4h_RSI_Range_HourFilter_VolumeSpike
Hypothesis: RSI mean reversion with hour filter (UTC 8-20) and volume spike. Works in both bull and bear by trading oversold/overbought extremes during active market hours. Uses RSI(14) with dynamic boundaries (30/70) and volume confirmation to reduce whipsaw. Targets 20-40 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Hour filter: UTC 8-20 (inclusive)
    hours = prices.index.hour
    hour_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Check filters
        if not (volume_filter[i] and hour_filter[i]):
            # Exit position if filters not met
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Long condition: RSI < 30 (oversold)
        if rsi[i] < 30:
            signals[i] = 0.25
            position = 1
        # Short condition: RSI > 70 (overbought)
        elif rsi[i] > 70:
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] > 40:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi[i] < 60:
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

name = "4h_RSI_Range_HourFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0