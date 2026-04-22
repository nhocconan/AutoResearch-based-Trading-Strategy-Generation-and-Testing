#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Daily data for Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray (standard period)
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray components to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_6h = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # 6h EMA20 for trend filter
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter - 60-period average
    vol_ma60 = pd.Series(prices['volume'].values).rolling(window=60, min_periods=60).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma60  # Moderate volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(ema_13_6h[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power (> 0) + price above EMA20 + volume surge
            if (bull_power_6h[i] > 0 and close[i] > ema_20[i] and vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power (< 0) + price below EMA20 + volume surge
            elif (bear_power_6h[i] < 0 and close[i] < ema_20[i] and vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power signals weaken or price crosses EMA13
            if position == 1:
                if bull_power_6h[i] <= 0 or close[i] < ema_13_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power_6h[i] >= 0 or close[i] > ema_13_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_MarketStructure_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0