#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h timeframe with 12h EMA50 trend filter and volume confirmation. 
In bull markets: buy when price breaks above Camarilla R3 and price > 12h EMA50. 
In bear markets: sell when price breaks below Camarilla S3 and price < 12h EMA50. 
Requires volume > 1.5x 20-period average for confirmation. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 50-150 total trades over 4 years = 12-37/year. 
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
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
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    hl_range_12h = high_12h - low_12h
    r3_12h = close_12h + (1.1 * hl_range_12h / 4)  # R3 = close + 1.1*(high-low)/4
    s3_12h = close_12h - (1.1 * hl_range_12h / 4)  # S3 = close - 1.1*(high-low)/4
    
    # Align Camarilla levels to match prices index
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R3 + 12h uptrend + volume confirmation
            long_setup = (close[i] > r3_aligned[i]) and htf_12h_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla S3 + 12h downtrend + volume confirmation
            short_setup = (close[i] < s3_aligned[i]) and htf_12h_bearish and volume_confirm
            
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
            # Exit: price touches Camarilla S3 (stop) OR 12h trend turns bearish
            if (close[i] <= s3_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R3 (stop) OR 12h trend turns bullish
            if (close[i] >= r3_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0