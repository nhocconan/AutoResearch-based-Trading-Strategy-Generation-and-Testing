#!/usr/bin/env python3
# 6h_12h_1d_volume_breakout_v1
# Hypothesis: Price breaking above/below 12h Donchian(20) channels with volume confirmation and 1-day trend filter.
# Long when price breaks above Donchian upper with volume > 1.5x 20-period average and 1d close > 1d SMA(50).
# Short when price breaks below Donchian lower with volume > 1.5x 20-period average and 1d close < 1d SMA(50).
# Uses 6h timeframe for entries and 12h/1d for trend/volume filters to reduce whipsaw.
# Designed for 12-37 trades/year on 6h timeframe. Works in bull markets via upside breakouts and bear markets via downside breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = high_roll.values
    donchian_low = low_roll.values
    
    # 6h volume MA(20) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # 12h SMA(50) for trend filter
    sma50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    sma50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma50_12h)
    
    # Get 1d data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d SMA(50) for trend filter
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure SMA(50) and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or \
           np.isnan(sma50_12h_aligned[i]) or np.isnan(sma50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower or trend reverses (12h or 1d)
            if close[i] < donchian_low[i] or close[i] < sma50_12h_aligned[i] or close[i] < sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper or trend reverses (12h or 1d)
            if close[i] > donchian_up[i] or close[i] > sma50_12h_aligned[i] or close[i] > sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper with volume surge and uptrend (both 12h and 1d)
            if close[i] > donchian_up[i] and vol_surge and close[i] > sma50_12h_aligned[i] and close[i] > sma50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower with volume surge and downtrend (both 12h and 1d)
            elif close[i] < donchian_low[i] and vol_surge and close[i] < sma50_12h_aligned[i] and close[i] < sma50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals