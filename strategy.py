#!/usr/bin/env python3
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
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian(20)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1d, highest_high_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above weekly EMA34 AND daily Donchian high with volume
            if close[i] > ema34_1w_aligned[i] and close[i] > donchian_high[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly EMA34 AND daily Donchian low with volume
            elif close[i] < ema34_1w_aligned[i] and close[i] < donchian_low[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below weekly EMA34 OR volume dries up
            if close[i] < ema34_1w_aligned[i] or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above weekly EMA34 OR volume dries up
            if close[i] > ema34_1w_aligned[i] or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyEMA34_DailyDonchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0