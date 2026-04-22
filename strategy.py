#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian channel breakout with 12-hour trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band and 12-hour EMA50 is rising with volume spike.
Short when price breaks below 20-period Donchian lower band and 12-hour EMA50 is falling with volume spike.
Exit when price returns to the Donchian midpoint or 12-hour EMA50 reverses.
Donchian channels provide clear breakout levels; 12-hour EMA filters for higher-timeframe trend direction;
volume spike confirms institutional participation. Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(dc_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above upper band with 12h EMA50 rising and volume spike
            if close[i] > dc_upper[i] and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with 12h EMA50 falling and volume spike
            elif close[i] < dc_lower[i] and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle band or 12h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle or 12h EMA50 turns down
                if close[i] <= dc_middle[i] or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle or 12h EMA50 turns up
                if close[i] >= dc_middle[i] or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_DonchianBreakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0