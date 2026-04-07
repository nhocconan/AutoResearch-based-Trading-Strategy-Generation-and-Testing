#!/usr/bin/env python3
"""
4h_donchian_breakout_12h_trend_volume_v3
Hypothesis: Breakouts from Donchian(15) channel on 4h filtered by 12-hour EMA20 trend and volume spike (>1.5x average).
Long when price breaks above upper Donchian with volume spike and price above 12h EMA20.
Short when price breaks below lower Donchian with volume spike and price below 12h EMA20.
Designed for ~25-35 trades/year on 4h with strict entry conditions to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    ema20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Donchian channels (15-period) on 4h
    high_max = pd.Series(high).rolling(window=15, min_periods=15).max().values
    low_min = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average (spike)
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > high_max[i-1]
        bearish_breakout = close[i] < low_min[i-1]
        
        # 12h trend filter
        above_12h_ema20 = close[i] > ema20_12h_aligned[i]
        below_12h_ema20 = close[i] < ema20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish breakout or trend turns bearish
            if bearish_breakout or below_12h_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish breakout or trend turns bullish
            if bullish_breakout or above_12h_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bullish Donchian breakout with volume spike and bullish trend
            if bullish_breakout and vol_spike and above_12h_ema20:
                position = 1
                signals[i] = 0.25
            # Short: bearish Donchian breakout with volume spike and bearish trend
            elif bearish_breakout and vol_spike and below_12h_ema20:
                position = -1
                signals[i] = -0.25
    
    return signals