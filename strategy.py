#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + Weekly EMA Trend Filter
Trades breakouts of 20-period Donchian channels on 12h timeframe.
Long when price breaks above upper band with volume spike and weekly EMA uptrend.
Short when price breaks below lower band with volume spike and weekly EMA downtrend.
Uses weekly EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency with clear trend-following edge in both bull and bear markets.
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
    
    # Get weekly data for EMA50 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_upper = price > high_20[i]
        below_lower = price < low_20[i]
        weekly_uptrend = price > ema_50_1w_aligned[i]
        weekly_downtrend = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: breakout above upper band, weekly uptrend, volume spike
            if above_upper and weekly_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band, weekly downtrend, volume spike
            elif below_lower and weekly_downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below lower band (contrary signal) or weekly trend turns down
            if below_lower or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above upper band (contrary signal) or weekly trend turns up
            if above_upper or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_WeeklyEMA50"
timeframe = "12h"
leverage = 1.0