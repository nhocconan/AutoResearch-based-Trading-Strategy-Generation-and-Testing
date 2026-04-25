#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
In bull markets: long when price breaks above Camarilla R1 and price > 12h EMA50.
In bear markets: short when price breaks below Camarilla S1 and price < 12h EMA50.
Requires volume > 2.0x 20-period average for confirmation (tight filter).
Exit on opposite Camarilla level touch or trend reversal.
Position size: 0.25 to limit drawdown.
Target: 75-200 total trades over 4 years = 19-50/year.
Uses 4h primary timeframe with 12h HTF for trend and volume context.
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
    
    # Get 12h data for HTF trend filter and volume context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation (on 12h)
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    hl_range_12h = high_12h - low_12h
    r1_12h = close_12h + (1.1 * hl_range_12h / 12)  # R1 = close + 1.1*(high-low)/12
    s1_12h = close_12h - (1.1 * hl_range_12h / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (tight filter)
        volume_confirm = volume[i] > 2.0 * vol_ma_20_12h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 12h uptrend + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_12h_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla S1 + 12h downtrend + volume confirmation
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
            # Exit: price touches Camarilla S1 (stop) OR 12h trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 12h trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0