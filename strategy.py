#!/usr/bin/env python3
"""
4h_donchian_breakout_12h_trend_volume_v1
Hypothesis: Donchian channel breakouts on 4h timeframe, when confirmed by volume
and aligned with 12h trend (EMA), capture strong momentum moves. Works in both bull
and bear markets by trading in the direction of the higher timeframe trend. 
Targets 20-50 trades per year by requiring confluence of breakout, volume surge,
and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: 20-period average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high + volume + uptrend
            if (close[i] > donchian_high[i] and 
                vol_confirm and 
                close[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short: price breaks below Donchian low + volume + downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.30
    
    return signals