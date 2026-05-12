#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike
# Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA34 trend filter and volume spike confirmation.
# Uses tight entry conditions to limit trades and avoid fee drag. Works in both bull and bear markets
# by following the 12h trend direction while using Camarilla levels for precise entries.
# Volume spike ensures institutional participation, reducing false breakouts.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for the given period.
    Returns R1, S1 levels.
    Formula:
    R1 = close + (high - low) * 1.1 / 12
    S1 = close - (high - low) * 1.1 / 12
    """
    R1 = close + (high - low) * 1.1 / 12
    S1 = close - (high - low) * 1.1 / 12
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume spike: current volume > 2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 to be valid
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 4h bar
        R1, S1 = calculate_camarilla(high[i], low[i], close[i])
        
        # Determine trend direction from 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above R1 AND uptrend on 12h AND volume spike
            if close[i] > R1 and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 AND downtrend on 12h AND volume spike
            elif close[i] < S1 and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR trend changes to downtrend
            if close[i] < S1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR trend changes to uptrend
            if close[i] > R1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals