#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Trade 6h breakouts at daily Camarilla R3/S3 levels with 12-hour EMA50 trend filter and volume confirmation.
R3/S3 levels represent strong intraday support/resistance where breakouts often continue with momentum.
In bull markets: buy when price breaks above daily R3 and price > 12h EMA50.
In bear markets: sell when price breaks below daily S3 and price < 12h EMA50.
Requires volume > 1.3x 24-period average for confirmation to filter weak breakouts.
Exit on opposite Camarilla level (R3/S3) touch or trend reversal.
Position size: 0.25 to limit drawdown.
Target: 50-150 total trades over 4 years = 12-37/year.
Daily Camarilla levels provide structure that works in both trending and ranging markets.
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
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter (12h timeframe)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 24-period average volume for confirmation (using 1d volume)
    volume_1d = df_1d['volume'].values
    vol_ma_24 = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    vol_ma_24_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_24)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    # Daily Camarilla R3 and S3 levels
    r3_1d = close_1d + (1.1 * hl_range_1d / 2)  # R3 = close + 1.1*(high-low)/2
    s3_1d = close_1d - (1.1 * hl_range_1d / 2)  # S3 = close - 1.1*(high-low)/2
    
    # Align daily Camarilla levels to 6h prices
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (24)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_24_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 24-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_24_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above daily Camarilla R3 + 12h uptrend + volume confirmation
            long_setup = (close[i] > r3_aligned[i]) and htf_12h_bullish and volume_confirm
            
            # Short setup: price breaks below daily Camarilla S3 + 12h downtrend + volume confirmation
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
            # Exit: price touches daily Camarilla S3 (stop) OR 12h trend turns bearish
            if (close[i] <= s3_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches daily Camarilla R3 (stop) OR 12h trend turns bullish
            if (close[i] >= r3_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0