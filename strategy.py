#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, weekly close > weekly open (bullish), and volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low, weekly close < weekly open (bearish), and volume > 1.5x 20-period average
# Exit when price retraces to Donchian midpoint or trend flips
# Uses weekly trend to avoid counter-trend trades in strong moves, works in both bull and bear via breakout logic
# Position size: 0.25 to limit drawdown, targeting 50-150 trades over 4 years

name = "6h_Donchian_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly trend filter: bullish if weekly close > weekly open
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    weekly_bullish = (df_1w['close'] > df_1w['open']).values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: breakout above Donchian high + weekly bullish + volume spike
            if (close[i] > donchian_high[i] and 
                weekly_bullish_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown below Donchian low + weekly bearish + volume spike
            elif (close[i] < donchian_low[i] and 
                  not weekly_bullish_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midpoint OR weekly trend turns bearish
            if (close[i] <= donchian_mid[i]) or (not weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement to midpoint OR weekly trend turns bullish
            if (close[i] >= donchian_mid[i]) or (weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals