#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Uses daily Camarilla R1/S1 levels for breakouts, filtered by 1d EMA trend and volume spike.
# Designed to capture strong moves in both bull and bear markets with tight entry conditions.
# Target: 15-25 trades/year per symbol with disciplined risk to avoid fee drag.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34 = close_1d_series.ewm(span=34, adjust=False).values
    
    # Calculate daily range for Camarilla levels
    daily_range = high_1d - low_1d
    camarilla_R1 = close_1d + daily_range * 1.1 / 12
    camarilla_S1 = close_1d - daily_range * 1.1 / 12
    
    # Align daily indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Camarilla R1 AND price above EMA(34) AND volume spike
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_aligned[i] and volume_ratio[i] > 2.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Camarilla S1 AND price below EMA(34) AND volume spike
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_aligned[i] and volume_ratio[i] > 2.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below Camarilla S1
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above Camarilla R1
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals