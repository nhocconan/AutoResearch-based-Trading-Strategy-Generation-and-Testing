#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with daily trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper channel (20-period) and price is above daily EMA50
# Short when price breaks below 4h Donchian lower channel (20-period) and price is below daily EMA50
# Volume must be above 1.5x 20-period average for confirmation
# Exit when price crosses back below/above the Donchian midpoint or trend changes
# Position size: 0.25 (25% of capital) to manage drawdown
# Designed to work in trending markets via daily EMA filter and range-bound markets via Donchian mean reversion

name = "4h_Donchian_DailyTrend_Volume"
timeframe = "4h"
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
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper AND above daily EMA50 + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower AND below daily EMA50 + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR trend turns bearish
            if (close[i] < donchian_mid[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR trend turns bullish
            if (close[i] > donchian_mid[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals