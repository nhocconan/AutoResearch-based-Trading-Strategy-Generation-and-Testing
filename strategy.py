#!/usr/bin/env python3
"""
4h_Donchian20_TrendFilter_VolumeSpike
Hypothesis: 4h Donchian channel (20) breakout with 12h EMA50 trend filter and volume confirmation (>2.0x 20-bar avg). 
Enters long when price breaks above upper band in 12h uptrend with volume spike, short when breaks below lower band in 12h downtrend with volume spike. 
Exits on opposite band touch or trend reversal. 
Designed for 4h timeframe with ~20-40 trades/year, works in bull/bear by following 12h trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period) using previous bar's high/low
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 20 bars for Donchian and 50 for EMA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band in 12h uptrend with volume confirmation
            long_setup = (close[i] > upper[i]) and (close[i] > ema_50_12h_aligned[i]) and volume_spike[i]
            # Short: price breaks below lower band in 12h downtrend with volume confirmation
            short_setup = (close[i] < lower[i]) and (close[i] < ema_50_12h_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches lower band OR trend turns down
            if (close[i] <= lower[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper band OR trend turns up
            if (close[i] >= upper[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_TrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0