#!/usr/bin/env python3
"""
12h_Donchian20_RangeBreakout_1dTrend_Volume
Hypothesis: Use 12h Donchian breakout above/below 20-bar channel with volume confirmation (>1.5x 20-bar avg) and filtered by 1d EMA50 trend. Go long on upward breakout with volume and price above 1d EMA50, short on downward breakout with volume and price below 1d EMA50. Exit on opposite Donchian breakout or trend reversal. Designed for 12h timeframe to limit trades (12-37/year) and avoid fee drift. Works in bull (catch breakouts) and bear (catch breakdowns) via symmetric logic.
"""

name = "12h_Donchian20_RangeBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel + volume spike + price above 1d EMA50
            if high[i] > high_20[i-1] and vol_spike and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower channel + volume spike + price below 1d EMA50
            elif low[i] < low_20[i-1] and vol_spike and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower channel or trend reversal (price below EMA50)
            if low[i] < low_20[i-1] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper channel or trend reversal (price above EMA50)
            if high[i] > high_20[i-1] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals