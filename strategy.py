#!/usr/bin/env python3
"""
4h_Alligator_AllLines_Cross_1dTrend_VolumeConfirm
Hypothesis: Williams Alligator (13,8,5 SMAs) with all lines aligned in same direction,
filtered by 1d EMA50 trend and volume spike confirmation. Works in bull/bear markets
by using 1d EMA50 as trend filter. Targets 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Alligator components on 4h data
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # Using SMA as approximation for SMMA (Williams uses SMMA but SMA is acceptable)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: all lines in same order
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Lips < Teeth < Jaw
        bullish_align = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_align = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend direction from 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = bullish_align and trend_up and vol_confirm
        short_entry = bearish_align and trend_down and vol_confirm
        
        # Exit conditions: Alligator lines cross or trend reversal
        # Exit long when lines cross bearish OR trend turns down
        long_exit = ((lips[i] < teeth[i]) or (teeth[i] < jaw[i])) or (not trend_up)
        # Exit short when lines cross bullish OR trend turns up
        short_exit = ((lips[i] > teeth[i]) or (teeth[i] > jaw[i])) or (not trend_down)
        
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

name = "4h_Alligator_AllLines_Cross_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0