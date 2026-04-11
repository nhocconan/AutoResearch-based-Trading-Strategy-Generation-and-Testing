#!/usr/bin/env python3
"""
6h_1w_Donchian_Breakout_Trend
Hypothesis: Uses weekly Donchian channels for trend direction and 6-hour Donchian breakouts with volume confirmation.
The weekly trend filters out counter-trend trades, while 6H breakouts capture momentum within the trend.
Volume confirmation ensures institutional participation. Works in bull/bear by following weekly trend.
Targets 15-30 trades/year per symbol with high-probability trend-following setups.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Donchian_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period) for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe (wait for weekly close)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # 6-hour Donchian breakout (20-period)
    donchian_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter: price relative to weekly Donchian
        weekly_uptrend = close[i] > donchian_high_20_aligned[i]  # Above weekly high = strong uptrend
        weekly_downtrend = close[i] < donchian_low_20_aligned[i]  # Below weekly low = strong downtrend
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # 6H breakout conditions
        breakout_6h_high = close[i] > donchian_high_6h[i]  # Break above 6H high
        breakdown_6h_low = close[i] < donchian_low_6h[i]   # Break below 6H low
        
        # Entry conditions: only trade in direction of weekly trend
        long_entry = breakout_6h_high and volume_filter and weekly_uptrend
        short_entry = breakdown_6h_low and volume_filter and weekly_downtrend
        
        # Exit conditions: opposite 6H breakout or trend change
        long_exit = breakdown_6h_low or (not weekly_uptrend)
        short_exit = breakout_6h_high or (not weekly_downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals