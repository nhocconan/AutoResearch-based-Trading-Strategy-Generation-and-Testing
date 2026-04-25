#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R3/S3 breakouts from 1-day levels on 4h timeframe with 1-day EMA34 trend filter and volume confirmation. 
R3/S3 levels provide strong intraday support/resistance. In bull markets: buy when price breaks above daily R3 and price > daily EMA34. 
In bear markets: sell when price breaks below daily S3 and price < daily EMA34. 
Requires volume > 1.8x 20-period average for confirmation to filter weak breakouts. 
Exit on opposite Camarilla level touch (R3/S3) or trend reversal. 
Position size: 0.30 to balance profit potential and drawdown control. 
Target: 100-200 total trades over 4 years = 25-50/year. 
Daily levels reduce noise vs weekly, improving win rate in both bull and bear markets with proper trend/volume filters.
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
    
    # Get 1d data for Camarilla levels, trend filter, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient daily data for EMA34
        return np.zeros(n)
    
    # Calculate daily EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation (using 1d volume)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    # Daily Camarilla R3 and S3 (strong breakout/breakdown levels)
    r3_1d = close_1d + (1.1 * hl_range_1d / 2)  # R3 = close + 1.1*(high-low)/2
    s3_1d = close_1d - (1.1 * hl_range_1d / 2)  # S3 = close - 1.1*(high-low)/2
    
    # Align daily Camarilla levels to 4h prices
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume[i] > 1.8 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above daily Camarilla R3 + 1d uptrend + volume confirmation
            long_setup = (close[i] > r3_aligned[i]) and htf_1d_bullish and volume_confirm
            
            # Short setup: price breaks below daily Camarilla S3 + 1d downtrend + volume confirmation
            short_setup = (close[i] < s3_aligned[i]) and htf_1d_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches daily Camarilla S3 (stop) OR 1d trend turns bearish
            if (close[i] <= s3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches daily Camarilla R3 (stop) OR 1d trend turns bullish
            if (close[i] >= r3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0