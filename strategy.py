#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Trade Camarilla pivot R1/S1 breakouts on daily timeframe with weekly volume confirmation.
Long when price breaks above weekly R1 with volume spike; short when breaks below weekly S1 with volume spike.
Uses weekly pivot levels for stronger support/resistance and volume filter to confirm institutional participation.
Designed for lower frequency (7-25 trades/year) to minimize fee drag and work in both bull/bear markets.
"""

name = "1d_1w_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot levels and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly volume average for spike detection
    vol_1w = df_1w['volume'].values
    vol_avg_1w = np.full(len(vol_1w), np.nan)
    for i in range(len(vol_1w)):
        if i >= 19:  # 20-period average
            vol_avg_1w[i] = np.mean(vol_1w[i-19:i+1])
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 5   # Need at least one week of prior data
    
    for i in range(start_idx, n):
        # Need at least one week of prior data for weekly Camarilla calculation
        if i < 5:   # Need 5 prior daily bars to have one week prior
            continue
            
        # Calculate weekly Camarilla levels using prior week's OHLC
        # Look back 5 days (1 week) to get prior week's data
        prior_week_high = np.max(df_1w['high'].iloc[:i]) if i < len(df_1w) else np.max(df_1w['high'].iloc[-1:])
        prior_week_low = np.min(df_1w['low'].iloc[:i]) if i < len(df_1w) else np.min(df_1w['low'].iloc[-1:])
        prior_week_close = df_1w['close'].iloc[i-1] if i-1 < len(df_1w) else df_1w['close'].iloc[-1]
        
        # Alternative: use aligned weekly data for current bar
        # Get the weekly data that corresponds to this daily bar
        week_idx = None
        for j in range(len(df_1w)):
            week_start = df_1w.index[j]
            if hasattr(week_start, 'to_pydatetime'):
                week_start = week_start.to_pydatetime()
            # Find which week this daily bar belongs to
            daily_date = pd.to_datetime(prices['open_time'].iloc[i])
            if hasattr(daily_date, 'to_pydatetime'):
                daily_date = daily_date.to_pydatetime()
            if j == 0 or (j > 0 and daily_date >= df_1w.index[j-1] and daily_date < df_1w.index[j]):
                week_idx = j-1 if j > 0 else 0
                break
        if week_idx is None:
            week_idx = len(df_1w) - 1
            
        if week_idx < 0 or week_idx >= len(df_1w):
            continue
            
        # Get prior week's data (completed week)
        prior_week_idx = week_idx - 1
        if prior_week_idx < 0:
            continue
            
        prior_week_high = df_1w['high'].iloc[prior_week_idx]
        prior_week_low = df_1w['low'].iloc[prior_week_idx]
        prior_week_close = df_1w['close'].iloc[prior_week_idx]
        
        # Calculate Camarilla levels
        range_val = prior_week_high - prior_week_low
        if range_val <= 0:
            continue
            
        # Camarilla R1 and S1 levels
        r1 = prior_week_close + (range_val * 1.1 / 12)
        s1 = prior_week_close - (range_val * 1.1 / 12)
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Volume spike: current volume > 1.5x weekly average volume
        vol_spike = (not np.isnan(vol_avg_1w_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_1w_aligned[i])
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike
            if current_close > r1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike
            elif current_close < s1 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly S1 or volume dries up significantly
            if current_close < s1 or (not np.isnan(vol_avg_1w_aligned[i]) and 
                                    current_volume < 0.3 * vol_avg_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly R1 or volume dries up significantly
            if current_close > r1 or (not np.isnan(vol_avg_1w_aligned[i]) and 
                                    current_volume < 0.3 * vol_avg_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals