#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
Uses 12h HTF for more stable trend identification vs 1d, reducing whipsaw in sideways markets.
Long: Close > R1 + price > 12h EMA50 + volume > 2.0 * 20-period average volume.
Short: Close < S1 + price < 12h EMA50 + volume > 2.0 * 20-period average volume.
Exit: Opposite Camarilla level touch OR trend reversal.
Position size: 0.25 (25% of capital) to balance profit potential and fee drag.
Target: 20-35 trades/year to stay well under the 400-trade 4h hard max.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    typical_price_12h = (h_12h + l_12h + c_12h) / 3.0
    range_12h = h_12h - l_12h
    camarilla_r1_12h = c_12h + (range_12h * 1.1 / 12.0)
    camarilla_s1_12h = c_12h - (range_12h * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (use previous 12h bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    
    # Volume confirmation: 4h volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 12h uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_12h_bullish and volume_spike[i]
            
            # Short setup: price breaks below Camarilla S1 + 12h downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_12h_bearish and volume_spike[i]
            
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
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 12h trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0