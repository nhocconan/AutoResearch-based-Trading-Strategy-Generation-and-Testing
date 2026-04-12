# This strategy uses 12h timeframe with daily Donchian breakouts and a 20-period EMA trend filter.
# It adds volume confirmation (current volume > 1.5x 20-period EMA volume) to filter breakouts.
# The strategy targets 20-50 trades per year by requiring multiple confirmations.
# It works in both bull and bear markets by following the daily trend and only taking breakouts
# in the direction of the trend with elevated volume.
# Risk management is built in via the trend filter - positions are closed when price crosses
# back below/above the EMA, preventing large drawdowns during reversals.

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
    
    # Get 1d data for trend filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d 20-period EMA (trend filter)
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 1d 20-period high and low for Donchian channels
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 20-period EMA of volume for volume filter
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ema20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period EMA volume
        vol_filter = volume[i] > vol_ema20[i] * 1.5
        
        # Trend filter: price above/below 1d 20 EMA
        price_above_ema20 = close[i] > ema20_1d_aligned[i]
        price_below_ema20 = close[i] < ema20_1d_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend with volume confirmation
        long_breakout = close[i] > high_20_aligned[i]  # break above 1d 20-period high
        short_breakout = close[i] < low_20_aligned[i]  # break below 1d 20-period low
        
        long_entry = long_breakout and price_above_ema20 and vol_filter
        short_entry = short_breakout and price_below_ema20 and vol_filter
        
        # Exit conditions: trend reversal (price crosses EMA)
        long_exit = close[i] < ema20_1d_aligned[i]
        short_exit = close[i] > ema20_1d_aligned[i]
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_ema20_vol_filter_v1"
timeframe = "12h"
leverage = 1.0