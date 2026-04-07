#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v2
Hypothesis: 4-hour Donchian channel breakouts with 1-day trend and volume filters capture sustained moves while avoiding whipsaws.
Works in bull markets by catching breakouts above upper band, and in bear markets by shorting breakdowns below lower band.
Volume confirms institutional participation, and 1-day EMA filter ensures alignment with higher timeframe trend.
Target: 20-40 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian Channel (20-period) on 4h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA for trend filter (20-period)
    ema_20_1d = df_1d['close'].ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend turns bearish
            if close[i] < low_roll[i] or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend turns bullish
            if close[i] > high_roll[i] or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with volume and bullish trend
            if (close[i] > high_roll[i] and vol_confirm and 
                close[i] > ema_20_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian with volume and bearish trend
            elif (close[i] < low_roll[i] and vol_confirm and 
                  close[i] < ema_20_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals