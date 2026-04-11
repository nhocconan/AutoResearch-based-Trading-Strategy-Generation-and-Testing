#!/usr/bin/env python3
# 6h_12h_donchian_breakout_volume_confirmation_v1
# Strategy: 6h Donchian(20) breakout with 12h volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture trend continuation. Volume confirmation from 12h timeframe
# filters false breakouts. Works in bull markets via upward breakouts and bear markets via downward
# breakouts. Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_breakout_volume_confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume average (20-period) for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Align raw 12h volume for confirmation
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_avg_20_12h_aligned[i]) or np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        vol_confirm = vol_12h_aligned[i] > 1.3 * vol_avg_20_12h_aligned[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]
        breakout_down = close[i] < donchian_low[i-1]
        
        # Entry conditions
        # Long: upward breakout with volume confirmation
        if breakout_up and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: downward breakout with volume confirmation
        elif breakout_down and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout (trend reversal)
        elif position == 1 and breakout_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals