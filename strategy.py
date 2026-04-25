#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Trade weekly Camarilla pivot (R4/S4) breakouts on 6h timeframe with 1-day EMA50 trend filter and volume confirmation. 
Weekly pivots provide stronger support/resistance than daily. In bull markets: buy when price breaks above weekly R4 and price > daily EMA50. 
In bear markets: sell when price breaks below weekly S4 and price < daily EMA50. 
Requires volume > 1.5x 20-period average for confirmation. 
Exit on opposite weekly Camarilla level touch or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 50-150 total trades over 4 years = 12-37/year. 
Weekly levels reduce noise and false breakouts, improving win rate in both bull and bear markets.
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
    
    # Get 1w data for weekly Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need sufficient weekly data
        return np.zeros(n)
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation (using 1d volume)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    hl_range_1w = high_1w - low_1w
    # Weekly Camarilla R4 and S4 (strongest breakout/breakdown levels)
    r4_1w = close_1w + (1.1 * hl_range_1w / 2)  # R4 = close + 1.1*(high-low)/2
    s4_1w = close_1w - (1.1 * hl_range_1w / 2)  # S4 = close - 1.1*(high-low)/2
    
    # Align weekly Camarilla levels to 6h prices
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above weekly Camarilla R4 + 1d uptrend + volume confirmation
            long_setup = (close[i] > r4_aligned[i]) and htf_1d_bullish and volume_confirm
            
            # Short setup: price breaks below weekly Camarilla S4 + 1d downtrend + volume confirmation
            short_setup = (close[i] < s4_aligned[i]) and htf_1d_bearish and volume_confirm
            
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
            # Exit: price touches weekly Camarilla S4 (stop) OR 1d trend turns bearish
            if (close[i] <= s4_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches weekly Camarilla R4 (stop) OR 1d trend turns bullish
            if (close[i] >= r4_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0