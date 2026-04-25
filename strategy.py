#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Volume_Spike_Trend_v1
Hypothesis: On daily timeframe, trade breakouts of Camarilla R3/S3 levels with 1-week EMA50 trend filter and volume spike confirmation. 
In bull markets: buy when price breaks above R3 and price > 1w EMA50. 
In bear markets: sell when price breaks below S3 and price < 1w EMA50. 
Requires volume > 2.5x 20-period average for confirmation to reduce false breakouts. 
Exit on opposite Camarilla level (R3/S3) touch or trend reversal. 
Position size: 0.25 to balance reward and risk and minimize fee churn. 
Target: 30-80 total trades over 4 years = 7-20/year (within 1d limits). 
Uses 1w HTF for more stable trend alignment than 1d, which should improve performance in both bull and bear markets.
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
    
    # Get 1d data for Camarilla levels and 1w data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for confirmation (on 1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    # Camarilla R3 and S3 (stronger resistance/support)
    r3_1d = close_1d + (1.1 * hl_range_1d / 4)  # R3 = close + 1.1*(high-low)/4
    s3_1d = close_1d - (1.1 * hl_range_1d / 4)  # S3 = close - 1.1*(high-low)/4
    
    # Align 1d Camarilla levels to 1d prices (same timeframe, no shift needed but use for consistency)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above 1w EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.5x 20-period average
        volume_confirm = volume[i] > 2.5 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above 1d Camarilla R3 + 1w uptrend + volume confirmation
            long_setup = (close[i] > r3_aligned[i]) and htf_1w_bullish and volume_confirm
            
            # Short setup: price breaks below 1d Camarilla S3 + 1w downtrend + volume confirmation
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
            # Exit: price touches 1d Camarilla S3 (stop) OR 1w trend turns bearish
            if (close[i] <= s3_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 1d Camarilla R3 (stop) OR 1w trend turns bullish
            if (close[i] >= r3_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Volume_Spike_Trend_v1"
timeframe = "1d"
leverage = 1.0