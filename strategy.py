#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Uses Elder Ray's Bull Power (high - EMA13) and Bear Power (EMA13 - low) on 1d timeframe,
# filtered by 1d EMA34 trend and volume spikes on 6h timeframe.
# In bull markets, Bull Power > 0 indicates buying pressure; in bear markets, Bear Power > 0 indicates selling pressure.
# The trend filter ensures we only take Elder Ray signals in the direction of the 1d EMA34 trend.
# Volume spikes confirm institutional participation. Designed for 6h to balance trade frequency and signal quality.
# Works in both bull and bear via trend-following Elder Ray signals.

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get 1d data for Elder Ray calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike on 6h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema_34_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + above 1d EMA34 trend + volume spike
            if bull_power_6h[i] > 0 and close[i] > ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 (selling pressure) + below 1d EMA34 trend + volume spike
            elif bear_power_6h[i] > 0 and close[i] < ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 (loss of buying pressure) or price below 1d EMA34
            if bull_power_6h[i] <= 0 or close[i] < ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power <= 0 (loss of selling pressure) or price above 1d EMA34
            if bear_power_6h[i] <= 0 or close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals