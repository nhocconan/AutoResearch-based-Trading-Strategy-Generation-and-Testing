#!/usr/bin/env python3
"""
4h_ThreeLeggedStool
4h strategy combining Donchian breakout, volume confirmation, and trend filter.
- Long: Close breaks above Donchian(20) high + volume > 1.5x 20-period average + EMA34 > EMA200
- Short: Close breaks below Donchian(20) low + volume > 1.5x 20-period average + EMA34 < EMA200
- Exit: Opposite breakout
Designed for ~20-50 trades/year per symbol (80-200 total over 4 years)
Uses price channel structure (Donchian) as primary signal, volume for confirmation,
and EMA trend filter to avoid counter-trend trades.
Works in both bull and bear markets by following the trend direction of breakouts.
"""

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
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA trend filter (34 and 200)
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_34[i]) or np.isnan(ema_200[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34[i] > ema_200[i]
        downtrend = ema_34[i] < ema_200[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > donch_high[i]
        breakdown_down = close[i] < donch_low[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above Donchian high
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below Donchian low
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: opposite breakout (below Donchian low)
            if breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite breakout (above Donchian high)
            if breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ThreeLeggedStool"
timeframe = "4h"
leverage = 1.0