#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
- Long when price breaks above 1d Camarilla R3 AND price > 1d EMA50 AND volume > 1.5x 24-period average
- Short when price breaks below 1d Camarilla S3 AND price < 1d EMA50 AND volume > 1.5x 24-period average
- Exit when price crosses the 1d Camarilla midpoint (R3/S3 average)
- Uses 1d Camarilla levels for structure and 1d EMA50 for HTF trend alignment
- Volume confirmation ensures institutional participation
- Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years)
- Works in both bull and bear markets: trend filter prevents counter-trend entries
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
    
    # Get 1d data for Camarilla levels and EMA50 (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d Camarilla levels
    # Camarilla: based on previous day's range
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shift by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous day
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_h5 = prev_close + 1.1 * range_1d / 2  # R3 equivalent
    camarilla_l5 = prev_close - 1.1 * range_1d / 2  # S3 equivalent
    camarilla_h6 = prev_close + 1.1 * range_1d      # R4
    camarilla_l6 = prev_close - 1.1 * range_1d      # S4
    camarilla_mid = (camarilla_h5 + camarilla_l5) / 2.0  # Midpoint of R3/S3
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume confirmation: > 1.5x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(25, 51, 25)  # Need 24 for volume MA, 50 for EMA50, 1 for Camarilla (but we use roll)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_h5_aligned[i]) or 
            np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 1d Camarilla R3/S3)
        breakout_up = close[i] > camarilla_h5_aligned[i]  # Break above R3
        breakout_down = close[i] < camarilla_l5_aligned[i]  # Break below S3
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 1d Camarilla midpoint (R3/S3 average)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < camarilla_mid_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > camarilla_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0