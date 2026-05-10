#!/usr/bin/env python3
# 4h_Chaikin_Oscillator_Trend_Filter
# Hypothesis: Chaikin Oscillator (3,10) crossing zero with EMA50 trend filter and volume confirmation.
# Long when Chaikin > 0 and price > EMA50; Short when Chaikin < 0 and price < EMA50.
# Volume confirmation ensures breakout strength. Works in bull/bear by following price trend.
# Targets 20-50 trades/year to minimize fee drag.

name = "4h_Chaikin_Oscillator_Trend_Filter"
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
    
    # Accumulation/Distribution Line
    clv = ((close - low) - (high - close)) / (high - low)
    clv = np.where((high - low) == 0, 0, clv)
    adl = np.cumsum(clv * volume)
    
    # Chaikin Oscillator: (3-day EMA of ADL) - (10-day EMA of ADL)
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # EMA50 trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(3, 10, 50, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(chaikin[i]) or np.isnan(ema50[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: Chaikin > 0 + uptrend + volume spike
            if chaikin[i] > 0 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Chaikin < 0 + downtrend + volume spike
            elif chaikin[i] < 0 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Chaikin crosses below zero or trend reversal
            if chaikin[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Chaikin crosses above zero or trend reversal
            if chaikin[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals