#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn_v1
Hypothesis: Trade 4h Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike confirmation. 
Camarilla R3/S3 provide stronger daily support/resistance than R1/S1, reducing false breakouts. 
In bull markets: buy when price breaks above R3 and price > 1d EMA34. 
In bear markets: sell when price breaks below S3 and price < 1d EMA34. 
Requires volume > 2.0x 20-period average for confirmation. 
Exit on opposite Camarilla level touch (R3/S3) or trend reversal. 
Position size: 0.25 to balance reward and risk and minimize fee churn. 
Target: 80-160 total trades over 4 years = 20-40/year. 
Uses 1d HTF for more stable trend alignment than 12h, improving performance in both bull and bear markets.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    # Camarilla R3 and S3 (stronger support/resistance levels)
    r3_1d = close_1d + (1.1 * hl_range_1d / 4)  # R3 = close + 1.1*(high-low)/4
    s3_1d = close_1d - (1.1 * hl_range_1d / 4)  # S3 = close - 1.1*(high-low)/4
    
    # Align 1d Camarilla levels to 4h prices
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above 1d EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above 1d Camarilla R3 + 1d uptrend + volume confirmation
            long_setup = (close[i] > r3_aligned[i]) and htf_1d_bullish and volume_confirm
            
            # Short setup: price breaks below 1d Camarilla S3 + 1d downtrend + volume confirmation
            short_setup = (close[i] < s3_aligned[i]) and htf_1d_bearish and volume_confirm
            
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
            # Exit: price touches 1d Camarilla S3 (stop) OR 1d trend turns bearish
            if (close[i] <= s3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 1d Camarilla R3 (stop) OR 1d trend turns bullish
            if (close[i] >= r3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn_v1"
timeframe = "4h"
leverage = 1.0