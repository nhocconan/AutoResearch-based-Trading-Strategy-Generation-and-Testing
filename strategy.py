#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTF12hTrend
Hypothesis: 4h Donchian(20) breakout with volume confirmation (>2.0x 20-bar avg) and 12h trend filter (price >/< 12h EMA34). 
Enters long on upper band breakout with volume spike in 12h uptrend, short on lower band breakout with volume spike in 12h downtrend. 
Exits on opposite Donchian breakout (short exit on upper breakout, long exit on lower breakout). 
Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe with ~20-40 trades/year, works in bull/bear by following 12h trend filter.
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
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian channels (20-period) on 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 20 bars for Donchian and EMA warmup
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike in 12h uptrend
            long_breakout = close[i] > highest_20[i-1]
            long_setup = long_breakout and volume_spike[i] and (close[i] > ema_34_12h_aligned[i])
            # Short: price breaks below lower Donchian band with volume spike in 12h downtrend
            short_breakout = close[i] < lowest_20[i-1]
            short_setup = short_breakout and volume_spike[i] and (close[i] < ema_34_12h_aligned[i])
            
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
            # Exit: price breaks below lower Donchian band (opposite breakout)
            if close[i] < lowest_20[i-1]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian band (opposite breakout)
            if close[i] > highest_20[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTF12hTrend"
timeframe = "4h"
leverage = 1.0