#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Daily Camarilla pivot levels (R1/S1) act as strong support/resistance. Price breaking above R1 with volume and daily uptrend = long signal. Breaking below S1 with volume and daily downtrend = short signal. Uses only 3 conditions (level, volume, trend) to keep trades low (~25/year). Works in bull (breaks R1) and bear (breaks S1) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day)
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # where C,H,L are from previous completed daily bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous, set to 0
    prev_close[0] = 0
    prev_high[0] = 0
    prev_low[0] = 0
    
    # Calculate Camarilla R1 and S1 for previous day
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe (these levels are valid for the entire day after the daily bar closes)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Daily trend filter: EMA34 on close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.8x 24-period average (4h * 6 = 24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma_24 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for EMA and volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Skip invalid Camarilla levels (first day)
        if R1_aligned[i] == 0 and S1_aligned[i] == 0:
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > R1_aligned[i]   # Break above R1
        breakout_short = close[i] < S1_aligned[i]  # Break below S1
        
        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_long and volume_spike[i] and uptrend
        short_entry = breakout_short and volume_spike[i] and downtrend
        
        # Exit on opposite breakout (only if volume confirms)
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0