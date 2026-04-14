#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d ADX trend filter and 1w Williams %R momentum
# ADX > 25 identifies strong trending conditions (works in both bull and bear markets)
# Williams %R identifies overbought/oversold conditions within the trend
# Long when: ADX > 25 (trending) + Williams %R crosses above -50 from below (bullish momentum)
# Short when: ADX > 25 (trending) + Williams %R crosses below -50 from above (bearish momentum)
# Uses 1d ADX for regime filter and 1w Williams %R for entry timing - avoids overtrading
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14 periods)
    adx_len = 14
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Directional Movement
    up = df_1d['high'] - df_1d['high'].shift(1)
    down = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    # Smoothed values
    tr_roll = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=adx_len, min_periods=adx_len).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=adx_len, min_periods=adx_len).sum()
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_roll
    minus_di = 100 * minus_dm_smooth / tr_roll
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Load 1w data ONCE for Williams %R
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Williams %R (14 periods)
    williams_len = 14
    highest_high = pd.Series(df_1w['high']).rolling(window=williams_len, min_periods=williams_len).max().values
    lowest_low = pd.Series(df_1w['low']).rolling(window=williams_len, min_periods=williams_len).min().values
    williams_r = (highest_high - df_1w['close'].values) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Handle division by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, adx_len * 2, williams_len * 2)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Entry conditions
            # Long: strong trend + Williams %R crosses above -50 from below
            long_entry = (strong_trend and 
                         williams_r_aligned[i] > -50 and 
                         i > start and 
                         williams_r_aligned[i-1] <= -50)
            # Short: strong trend + Williams %R crosses below -50 from above
            short_entry = (strong_trend and 
                          williams_r_aligned[i] < -50 and 
                          i > start and 
                          williams_r_aligned[i-1] >= -50)
            
            if long_entry:
                position = 1
                signals[i] = position_size
            elif short_entry:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens OR momentum reverses
            exit_long = (adx_aligned[i] <= 25) or \
                       (williams_r_aligned[i] < -80)  # Oversold exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakens OR momentum reverses
            exit_short = (adx_aligned[i] <= 25) or \
                        (williams_r_aligned[i] > -20)  # Overbought exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dADX_1wWilliamsR_Trend_Momentum_v1"
timeframe = "6h"
leverage = 1.0