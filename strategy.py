#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Trade daily Camarilla R1/S1 breakouts with 1-week EMA50 trend filter and volume spike confirmation. 
Camarilla R1/S1 provide key daily support/resistance levels derived from previous day's range. 
In bull markets (price > 1w EMA50): buy when price breaks above R1 with volume > 2.0x 20-day average. 
In bear markets (price < 1w EMA50): sell when price breaks below S1 with volume > 2.0x 20-day average. 
Exit on opposite Camarilla level touch or trend reversal. 
Position size: 0.25 to balance reward and risk and minimize fee churn. 
Target: 30-100 total trades over 4 years = 7-25/year (within 1d optimal range). 
Uses 1w HTF for more stable trend alignment vs shorter HTF, which should improve performance in both bull and bear markets by reducing whipsaw.
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume for confirmation (using 1d data resampled to 1d frequency)
    # Since we're on 1d timeframe, volume is already daily
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels use previous day's range
    hl_range_1d = high_1d - low_1d
    # Daily Camarilla R1 and S1 (key intraday resistance/support)
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Since we're on 1d timeframe, no alignment needed - values are already per bar
    r1_aligned = r1_1d
    s1_aligned = s1_1d
    ema_50_1w_aligned = ema_50_1w  # Already aligned via align_htf_to_ltf
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above 1w EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above daily Camarilla R1 + 1w uptrend + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_1w_bullish and volume_confirm
            
            # Short setup: price breaks below daily Camarilla S1 + 1w downtrend + volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and htf_1w_bearish and volume_confirm
            
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
            # Exit: price touches daily Camarilla S1 (stop) OR 1w trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches daily Camarilla R1 (stop) OR 1w trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0