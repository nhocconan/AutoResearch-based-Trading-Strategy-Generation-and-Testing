#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeConfirm_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h timeframe with 4h EMA20 trend filter and 1d volume spike confirmation.
In bull markets: buy when price breaks above Camarilla R1 and price > 4h EMA20.
In bear markets: sell when price breaks below Camarilla S1 and price < 4h EMA20.
Requires 1d volume > 2.0x 20-period average for confirmation (avoid low-volume breakouts).
Exit on opposite Camarilla level touch or 4h trend reversal.
Position size: 0.20 to limit drawdown and enable discrete sizing.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
Uses 4h for signal direction, 1d for volume regime filter, 1h only for entry timing precision.
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
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for HTF trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for Camarilla levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA20 (20) and 1d volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend (bullish = price above 4h EMA20)
        htf_4h_bullish = close[i] > ema_20_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_20_4h_aligned[i]
        
        # Volume confirmation: 1d volume > 2.0x 20-period average (avoid low-volume breakouts)
        volume_confirm = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 4h uptrend + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_4h_bullish and volume_confirm
            
            # Short setup: price breaks below Camarilla S1 + 4h downtrend + volume confirmation
            short_setup = (close[i] < s1_aligned[i]) and htf_4h_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price touches Camarilla S1 (stop) OR 4h trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_4h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price touches Camarilla R1 (stop) OR 4h trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_4h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0