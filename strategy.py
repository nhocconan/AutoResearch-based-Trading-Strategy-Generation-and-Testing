#!/usr/bin/env python3
# 4h_Combined_Camarilla_Donchian_Breakout_1dTrend_Volume
# Hypothesis: Combine strengths of Camarilla R1/S1 and Donchian(20) breakouts with 1d EMA34 trend filter and volume confirmation (2x 20-period avg). This dual-trigger approach increases signal reliability while maintaining low trade frequency. Works in bull markets via breakouts and bear markets via mean-reversion at extreme levels. Target: 25-40 trades/year.

name = "4h_Combined_Camarilla_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Load 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1_level = prev_close_1d + 1.1 * camarilla_range * 1.0 / 4  # R1 = C + 1.1*range/4
    s1_level = prev_close_1d - 1.1 * camarilla_range * 1.0 / 4  # S1 = C - 1.1*range/4
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Calculate Donchian channels (20-period) from 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_current = close_1d_aligned[i]
        
        trend_up = close_1d_current > ema34_1d_aligned[i]
        trend_down = close_1d_current < ema34_1d_aligned[i]
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: Either Camarilla R1 breakout OR Donchian upper breakout with uptrend and volume
            long_condition = ((close[i] > r1_aligned[i]) or (close[i] > high_20[i])) and trend_up and vol_confirm
            # SHORT: Either Camarilla S1 breakdown OR Donchian lower breakdown with downtrend and volume
            short_condition = ((close[i] < s1_aligned[i]) or (close[i] < low_20[i])) and trend_down and vol_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low OR Camarilla S1
            if close[i] < low_20[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high OR Camarilla R1
            if close[i] > high_20[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals