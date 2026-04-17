#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w EMA34 trend filter + 1d Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day high with weekly EMA34 > prior weekly EMA34 (uptrend) and volume > 1.5x 20-day volume average.
Short when price breaks below 20-day low with weekly EMA34 < prior weekly EMA34 (downtrend) and volume > 1.5x 20-day volume average.
Weekly EMA34 provides reliable trend filter to avoid counter-trend breakouts. Designed for low trade frequency (<25/year) with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian(20) channels
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Weekly trend: current EMA34 > previous EMA34 (uptrend) or < (downtrend)
        # Need previous aligned weekly value for comparison
        if i > 0 and not np.isnan(ema_34_1w_aligned[i-1]):
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            weekly_downtrend = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        else:
            weekly_uptrend = False
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above 20-day high with weekly uptrend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                weekly_uptrend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with weekly downtrend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  weekly_downtrend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wEMA34_Trend_Donchian20_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0