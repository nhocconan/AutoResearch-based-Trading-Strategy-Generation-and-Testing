#!/usr/bin/env python3
# 101017
# 4h_Donchian20_Breakout_1dTrend_VolumeS
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses daily trend to filter breakout direction (long only in uptrend, short only in downtrend).
# Volume spike ensures breakout has conviction. Target 20-50 trades/year to minimize fee drag.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Avoids false breakouts in ranging markets via trend filter.

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h Donchian(20) channel
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = np.roll(high_roll, 1)
    lower_band = np.roll(low_roll, 1)
    upper_band[0] = np.nan
    lower_band[0] = np.nan
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper band, above EMA34 (uptrend), volume spike
        if (close[i] > upper_band[i] and 
            close[i] > ema34_1d_aligned[i] and 
            volume_spike[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower band, below EMA34 (downtrend), volume spike
        elif (close[i] < lower_band[i] and 
              close[i] < ema34_1d_aligned[i] and 
              volume_spike[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite band (mean reversion)
        elif position == 1 and close[i] < lower_band[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > upper_band[i]:
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

name = "4h_Donchian20_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0