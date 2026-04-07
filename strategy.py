#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: Breakouts from Donchian(20) channel on 12h filtered by 1-day EMA25 trend and volume spike (>1.8x average).
Long when price breaks above upper Donchian with volume spike and price above 1d EMA25.
Short when price breaks below lower Donchian with volume spike and price below 1d EMA25.
Designed for ~15-25 trades/year on 12h with strict entry conditions to avoid overtrading and capture trend in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    ema25_1d = pd.Series(df_1d['close'].values).ewm(span=25, adjust=False).mean().values
    ema25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    
    # Donchian channels (20-period) on 12h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 24-period average (2 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(ema25_1d_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x average (spike)
        vol_spike = volume[i] > (vol_ma[i] * 1.8)
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > high_max[i-1]
        bearish_breakout = close[i] < low_min[i-1]
        
        # 1d trend filter
        above_1d_ema25 = close[i] > ema25_1d_aligned[i]
        below_1d_ema25 = close[i] < ema25_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish breakout or trend turns bearish
            if bearish_breakout or below_1d_ema25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish breakout or trend turns bullish
            if bullish_breakout or above_1d_ema25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bullish Donchian breakout with volume spike and bullish trend
            if bullish_breakout and vol_spike and above_1d_ema25:
                position = 1
                signals[i] = 0.25
            # Short: bearish Donchian breakout with volume spike and bearish trend
            elif bearish_breakout and vol_spike and below_1d_ema25:
                position = -1
                signals[i] = -0.25
    
    return signals