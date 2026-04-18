#!/usr/bin/env python3
"""
6h_Weekly_Donchian_Breakout_With_Volume_Filter
Hypothesis: On 6h timeframe, buy when price breaks above weekly Donchian high (20-period) with volume confirmation, sell when price breaks below weekly Donchian low. Weekly trend filter ensures we only trade in direction of weekly trend (price above/below weekly SMA50). This captures major trend continuations while avoiding counter-trend breakouts. Works in bull markets by catching breakouts and in bear markets by avoiding false breakouts via trend filter. Targets ~20-30 trades/year by requiring weekly Donchian breakout + volume + trend filter.
"""

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
    
    # Get weekly data for Donchian channels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    weekly_donch_high = np.full_like(weekly_close, np.nan)
    weekly_donch_low = np.full_like(weekly_close, np.nan)
    
    for i in range(20, len(weekly_close)):
        weekly_donch_high[i] = np.max(weekly_high[i-20:i])
        weekly_donch_low[i] = np.min(weekly_low[i-20:i])
    
    # Calculate weekly SMA50 for trend filter
    weekly_sma50 = np.full_like(weekly_close, np.nan)
    for i in range(50, len(weekly_close)):
        weekly_sma50[i] = np.mean(weekly_close[i-50:i])
    
    # Align weekly indicators to 6h timeframe
    weekly_donch_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donch_high)
    weekly_donch_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_donch_low)
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_sma50)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need weekly SMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_donch_high_aligned[i]) or np.isnan(weekly_donch_low_aligned[i]) or
            np.isnan(weekly_sma50_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + volume + weekly uptrend
            if (high[i] > weekly_donch_high_aligned[i] and 
                volume_confirm and 
                weekly_sma50_aligned[i] < weekly_close[-1] if len(weekly_close) > 0 else True):  # weekly trend up (price above SMA50)
                # Simplified trend filter: weekly price above weekly SMA50
                if not np.isnan(weekly_sma50_aligned[i]) and weekly_close[-1] > weekly_sma50_aligned[-1] if len(weekly_close) > 0 else True:
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below weekly Donchian low + volume + weekly downtrend
            elif (low[i] < weekly_donch_low_aligned[i] and 
                  volume_confirm and 
                  not np.isnan(weekly_sma50_aligned[i]) and weekly_close[-1] < weekly_sma50_aligned[-1] if len(weekly_close) > 0 else True):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low
            if low[i] < weekly_donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high
            if high[i] > weekly_donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Donchian_Breakout_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0