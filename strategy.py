#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSlope
Hypothesis: Combines 12h EMA50 trend filter with daily Camarilla R3/S3 breakout. 
Uses volume slope (rising volume) as confirmation to avoid fakeouts. 
Designed for low trade frequency (<25/year) to minimize fee burn while capturing 
strong directional moves in both bull and bear markets by requiring alignment 
across multiple timeframes and volume confirmation.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R3 = typical_price + (range_hl * 1.1 / 2)
    S3 = typical_price - (range_hl * 1.1 / 2)
    
    # Align to lower timeframe (4h) - values from previous day's close
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    typical_price_aligned = align_htf_to_ltf(prices, df_1d, typical_price.values)
    
    # Volume slope confirmation: current volume > previous volume (rising volume)
    volume_slope = volume > np.roll(volume, 1)
    volume_slope[0] = False  # First value has no previous
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(typical_price_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume slope confirmation: rising volume
        vol_confirm = volume_slope[i]
        
        # Entry conditions: Camarilla breakout with volume slope and trend alignment
        long_entry = (close[i] > R3_aligned[i]) and vol_confirm and uptrend
        short_entry = (close[i] < S3_aligned[i]) and vol_confirm and downtrend
        
        # Exit conditions: price returns to typical price level or trend reverses
        long_exit = close[i] < typical_price_aligned[i]
        short_exit = close[i] > typical_price_aligned[i]
        
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

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSlope"
timeframe = "4h"
leverage = 1.0