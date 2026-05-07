#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_Filter
Hypothesis: Elder Ray (Bull/Bear Power) captures institutional buying/selling pressure. 
In bull markets: enter long when Bull Power > 0 and Bear Power < 0 with 1-day EMA uptrend.
In bear markets: enter short when Bear Power < 0 and Bull Power > 0 with 1-day EMA downtrend.
Adds volume confirmation to avoid false signals. Designed for both bull and bear regimes.
Target: 15-25 trades/year to minimize fee drag.
"""

name = "6h_ElderRay_BullBearPower_1dTrend_Filter"
timeframe = "6h"
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
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying pressure
    bear_power = low - ema13   # Selling pressure
    
    # 1-day EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if any critical value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive, Bear Power negative, 1-day uptrend, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, Bull Power positive, 1-day downtrend, volume spike
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  close[i] < ema_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative OR price breaks below 1-day EMA
            if bull_power[i] <= 0 or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive OR price breaks above 1-day EMA
            if bear_power[i] >= 0 or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals