#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Camarilla R1/S1 breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above 4h Camarilla R1 with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below 4h Camarilla S1 with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the 4h Camarilla midpoint (mean reversion to pivot center).
Uses 4h for structure (Camarilla pivots from actual 4h bars) and 1d for trend filter to avoid counter-trend trades.
Session filter (08-20 UTC) reduces noise trades. Target position size 0.20.
Designed to work in both bull and bear markets by combining mean-reversion pivots with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (R1, S1, midpoint) from previous 4h bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using previous completed 4h bar to avoid look-ahead
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = np.nan  # first bar has no previous
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    camarilla_range = prev_high_4h - prev_low_4h
    r1_4h = prev_close_4h + 1.1 * camarilla_range / 12.0
    s1_4h = prev_close_4h - 1.1 * camarilla_range / 12.0
    midpoint_4h = (r1_4h + s1_4h) / 2.0  # equals prev_close_4h
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1h volume 24-period average for confirmation (approx 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or 
            np.isnan(midpoint_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 24-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_24[i]
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R1 with volume and uptrend (price > EMA50)
            if (close[i] > r1_4h_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S1 with volume and downtrend (price < EMA50)
            elif (close[i] < s1_4h_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 4h Camarilla midpoint
            if close[i] <= midpoint_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to or above 4h Camarilla midpoint
            if close[i] >= midpoint_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hCamarilla_R1S1_Breakout_Volume_1dEMA50_Session"
timeframe = "1h"
leverage = 1.0