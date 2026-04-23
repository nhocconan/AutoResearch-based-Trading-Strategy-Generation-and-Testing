#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation.
- Primary timeframe: 12h for lower trade frequency (target: 12-37/year).
- Donchian breakout captures medium-term trends; EMA34 filter aligns with higher timeframe trend.
- Volume confirmation ensures breakouts have participation.
- Discrete position sizing (±0.30) to minimize fee churn.
- Works in bull markets (long breakouts above EMA34) and bear markets (short breakdowns below EMA34).
- Exit on opposite Donchian breakout or trend reversal.
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + above 1d EMA34 + volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below Donchian low + below 1d EMA34 + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR closes below 1d EMA34
            if close[i] < donchian_low[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR closes above 1d EMA34
            if close[i] > donchian_high[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0