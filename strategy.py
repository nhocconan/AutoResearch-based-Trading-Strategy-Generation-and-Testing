#!/usr/bin/env python3
# 12h_1w_donchian_volume_trend_v1
# Hypothesis: Trade 12h Donchian channel breakouts with 1w trend filter and volume confirmation.
# Works in bull markets (breakouts above upper band with uptrend) and bear markets (breakouts below lower band with downtrend).
# Uses 1w EMA50 for trend, 12h Donchian(20) for breakout, and volume surge for confirmation.
# Target: 15-30 trades/year on 12h timeframe to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 12h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 12h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price below lower Donchian band OR trend reversal
            if close[i] < low_roll[i] or ema50_1w_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above upper Donchian band OR trend reversal
            if close[i] > high_roll[i] or ema50_1w_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above upper Donchian band with uptrend and volume surge
            if close[i] > high_roll[i] and ema50_1w_aligned[i] < close[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price below lower Donchian band with downtrend and volume surge
            elif close[i] < low_roll[i] and ema50_1w_aligned[i] > close[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals