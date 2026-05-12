#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_12H_TREND_FILTER
# Hypothesis: 4-hour Donchian channel (20-period) breakouts with volume confirmation
# and 12-hour EMA50 trend filter capture strong momentum moves. Works in bull markets
# (breakouts continuation) and bear markets (mean reversion at extremes via tight stops).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_12H_TREND_FILTER"
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
    
    # Calculate 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # 12-hour EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian with volume confirmation in uptrend
            if (close[i] > high_roll[i] and 
                vol_confirm[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian with volume confirmation in downtrend
            elif (close[i] < low_roll[i] and 
                  vol_confirm[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below lower Donchian or trend reversal
            if (close[i] < low_roll[i] or 
                close[i] <= ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above upper Donchian or trend reversal
            if (close[i] > high_roll[i] or 
                close[i] >= ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals