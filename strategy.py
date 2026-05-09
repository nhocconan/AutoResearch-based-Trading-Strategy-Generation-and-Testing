#!/usr/bin/env python3
# Hypothesis: 6h Donchian channel breakout with weekly trend filter and volume confirmation
# Long when price breaks above 6h Donchian(20) high, weekly trend is up (price > weekly EMA50), and volume > 1.5x average
# Short when price breaks below 6h Donchian(20) low, weekly trend is down (price < weekly EMA50), and volume > 1.5x average
# Exit when price returns to the 6h Donchian midpoint
# Uses Donchian for breakout structure, weekly EMA for trend filter, volume for conviction
# Designed to capture strong trending moves while avoiding choppy markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_Donchian_WeeklyTrend_Volume"
timeframe = "6h"
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
    
    # Calculate 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().values
    donchian_low = low_roll.min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate weekly trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, weekly trend up, volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1w_aligned[i] and  # Price above weekly EMA50 (uptrend)
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, weekly trend down, volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1w_aligned[i] and  # Price below weekly EMA50 (downtrend)
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals