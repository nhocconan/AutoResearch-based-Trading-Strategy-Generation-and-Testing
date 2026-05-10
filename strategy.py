#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakouts filtered by 4h EMA50 trend and volume surge.
# Uses 4h for signal direction, 1h for entry timing to reduce false breakouts.
# Targets 15-37 trades/year (60-150 total) to minimize fee drag.
# Session filter (08-20 UTC) reduces noise. Position size 0.20.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 4h data for Camarilla calculation (use previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    camarilla_multiplier = 1.0833  # for R1/S1
    high_low_range = high_4h - low_4h
    
    r1 = close_4h_vals + (high_low_range * camarilla_multiplier)
    s1 = close_4h_vals - (high_low_range * camarilla_multiplier)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # 1h price data
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Volume average (24-period)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need EMA50 (50) + volume MA (24)
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals (using previous bar's levels)
        breakout_r1 = close[i] > r1_aligned[i-1]
        breakdown_s1 = close[i] < s1_aligned[i-1]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Camarilla R1 breakout with volume surge and 4h uptrend
            if breakout_r1 and volume_surge and uptrend and in_session[i]:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla S1 breakdown with volume surge and 4h downtrend
            elif breakdown_s1 and volume_surge and downtrend and in_session[i]:
                signals[i] = -0.20
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 4 bars
            if bars_since_entry < 4:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below S1 or trend changes
                if close[i] < s1_aligned[i-1] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: price breaks above R1 or trend changes
                if close[i] > r1_aligned[i-1] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals