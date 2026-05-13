#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Trend_Filter_Volume_Spike
Hypothesis: Donchian channel breakouts from 12h data, filtered by weekly trend and daily volume spikes, capture strong directional moves with low trade frequency. Designed to work in both bull and bear markets by following the weekly trend direction only, reducing false breakouts during ranging periods.
"""

name = "12h_Donchian_Breakout_Trend_Filter_Volume_Spike"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for volume confirmation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day average volume for spike detection
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Volume spike: current volume > 2.0x daily average volume
        volume_spike = volume[i] > (2.0 * vol_avg_1d_aligned[i])
        
        if position == 0:
            # LONG: break above Donchian high with volume spike and above weekly EMA50 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_spike and 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: break below Donchian low with volume spike and below weekly EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_spike and 
                  close[i] < trend_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price drops below Donchian low or trend turns down
            if (close[i] < donchian_low[i] or 
                close[i] < trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above Donchian high or trend turns up
            if (close[i] > donchian_high[i] or 
                close[i] > trend_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals