#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 12h timeframe with 1-week EMA34 trend filter and volume confirmation. 
In bull markets: buy when price breaks above Camarilla R3 and price > weekly EMA34. 
In bear markets: sell when price breaks below Camarilla S3 and price < weekly EMA34. 
Requires volume > 1.5x 20-period average for confirmation. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 50-150 total trades over 4 years = 12-37/year. 
Uses weekly HTF for stronger trend filter, reducing whipsaw in ranging markets.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate weekly EMA34 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla levels (more stable than calculating on 12h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    
    # Calculate 20-period average volume for confirmation (using 1d volume)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    r3_1d = close_1d + (1.1 * hl_range_1d / 4)  # R3 = close + 1.1*(high-low)/4
    s3_1d = close_1d - (1.1 * hl_range_1d / 4)  # S3 = close - 1.1*(high-low)/4
    
    # Align Camarilla levels to match prices index
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above weekly EMA34)
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R3 + 1w uptrend + volume confirmation
            long_setup = (close[i] > r3_aligned[i]) and htf_1w_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla S3 + 1w downtrend + volume confirmation
            short_setup = (close[i] < s3_aligned[i]) and htf_1w_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla S3 (stop) OR 1w trend turns bearish
            if (close[i] <= s3_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R3 (stop) OR 1w trend turns bullish
            if (close[i] >= r3_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0