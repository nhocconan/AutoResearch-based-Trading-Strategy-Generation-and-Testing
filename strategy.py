#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Camarilla pivot breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
Goes long when price breaks above R1 in uptrend, short when breaks below S1 in downtrend.
Uses volume spike (>2x 20-bar MA) to confirm breakouts. Designed for low trade frequency
(20-30 trades/year) to minimize fee drag while capturing strong directional moves.
Works in both bull and bear markets by following 12h trend direction.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    # We'll calculate daily OHLC from the price data by resampling conceptually
    # but using actual daily bars from data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas:
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # R2 = close + 0.6 * (high - low)
    # R1 = close + 0.4 * (high - low)
    # PP = (high + low + close) / 3
    # S1 = close - 0.4 * (high - low)
    # S2 = close - 0.6 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    
    camarilla_r1 = close_1d + 0.4 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 0.4 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: >2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions
        breakout_r1 = close[i] > camarilla_r1_aligned[i]
        breakdown_s1 = close[i] < camarilla_s1_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_r1
        short_entry = vol_confirm and downtrend and breakdown_s1
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakdown_s1 or (not uptrend)
        short_exit = breakout_r1 or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0