#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-week trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band, 1w trend is up, and volume spike.
Short when price breaks below 20-period Donchian lower band, 1w trend is down, and volume spike.
Exit when price crosses the midline or trend reverses.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1w close for trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_30[i]) or
            np.isnan(donch_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0:
            # Long: break above upper band, 1w trend up, volume spike
            if (close[i] > donch_high[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band, 1w trend down, volume spike
            elif (close[i] < donch_low[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses midline or 1w trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below midline or 1w trend turns down
                if close[i] < donch_mid[i] or ema20_1w_aligned[i] < ema20_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above midline or 1w trend turns up
                if close[i] > donch_mid[i] or ema20_1w_aligned[i] > ema20_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0