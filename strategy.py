#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Uses 12h primary timeframe for low trade frequency (~12-37/year expected).
# Camarilla R3/S3 levels from 1d provide institutional pivot points with proven edge.
# 1d EMA50 filters for trend alignment, reducing counter-trend trades in bear markets.
# Volume spike (>2.0x 20-bar average) confirms breakout strength and filters low-momentum false breakouts.
# Position size 0.25 for balanced risk/return. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0  # R3 = pivot + 1.1*(range)/2
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0  # S3 = pivot - 1.1*(range)/2
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 direction
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > r3_1d_aligned[i]
        short_breakout = close[i] < s3_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level (R2/S2 for faster exit)
        # Calculate R2/S2 for exit
        r2_1d = pivot_1d + range_1d * 1.1 / 4.0  # R2 = pivot + 1.1*(range)/4
        s2_1d = pivot_1d - range_1d * 1.1 / 4.0  # S2 = pivot - 1.1*(range)/4
        r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
        s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
        
        long_exit = close[i] < s2_1d_aligned[i]  # Exit long at S2
        short_exit = close[i] > r2_1d_aligned[i]  # Exit short at R2
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals