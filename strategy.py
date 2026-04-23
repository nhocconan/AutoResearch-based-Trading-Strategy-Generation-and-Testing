#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation.
- Donchian: Upper = 20-period high, Lower = 20-period low
- Long: Close breaks above Upper Band AND price > 1w EMA50 AND volume > 2.0x 20-period avg
- Short: Close breaks below Lower Band AND price < 1w EMA50 AND volume > 2.0x 20-period avg
- Exit: Close crosses 12h EMA34 (middle band proxy) OR Donchian band reversal
- Works in bull (buy breakouts) and bear (sell breakdowns) with trend filter
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA34 for exit/middle band
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 34)  # Need 50 for 1w EMA50, 20 for Donchian, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(ema_34[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Donchian Upper Band AND price > 1w EMA50
            if (close[i] > donch_high[i] and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian Lower Band AND price < 1w EMA50
            elif (close[i] < donch_low[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below 12h EMA34 OR Donchian Lower Band (reversal)
            if close[i] < ema_34[i] or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above 12h EMA34 OR Donchian Upper Band (reversal)
            if close[i] > ema_34[i] or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0