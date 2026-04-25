#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v3
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume spike confirmation. 
Camarilla R1/S1 provide daily support/resistance levels. In bull markets: buy when price breaks above R1 and price > 12h EMA50. 
In bear markets: sell when price breaks below S1 and price < 12h EMA50. 
Requires volume > 2.2x 20-period average for confirmation to reduce false breakouts and overtrading. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to balance reward and risk and reduce fee churn. 
Target: 60-120 total trades over 4 years = 15-30/year (reduced from v2 to avoid overtrading). 
Uses 12h HTF for more stable trend alignment than 1d, which should improve performance in both bull and bear markets.
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
    
    # Get 12h data for HTF trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels from 12h data (approximation)
    # Using 12h high/low/close as proxy for daily levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    hl_range_12h = high_12h - low_12h
    # Daily Camarilla R1 and S1 (key intraday resistance/support) using 12h data
    r1_12h = close_12h + (1.1 * hl_range_12h / 12)  # R1 = close + 1.1*(high-low)/12
    s1_12h = close_12h - (1.1 * hl_range_12h / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align 12h Camarilla levels to 4h prices
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.2x 20-period average (tighter than v2's 2.0x)
        volume_confirm = volume[i] > 2.2 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above 12h Camarilla R1 + 12h uptrend + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_12h_bullish and volume_confirm
            
            # Short setup: price breaks below 12h Camarilla S1 + 12h downtrend + volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and htf_12h_bearish and volume_confirm
            
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
            # Exit: price touches 12h Camarilla S1 (stop) OR 12h trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 12h Camarilla R1 (stop) OR 12h trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0