#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d trend filter and volume confirmation capture institutional level breaks with lower frequency than R4/S4. 
In bull markets: price breaks above R3 (strong resistance) with 1d uptrend and volume spike → long. 
In bear markets: price breaks below S3 (strong support) with 1d downtrend and volume spike → short. 
Uses 1d EMA34 for trend (more stable than 12h) and volume > 1.5x 20-period median for confirmation. 
Target: 50-150 total trades over 4 years (12-37/year). Camarilla pivots from 12h provide structure, 1d trend filters noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need 20 for volume median
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period median (less strict than 2.0x to allow more trades)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 1.5)
    
    # Load 12h data for Camarilla pivots (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar's OHLC (R3/S3)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    typical_price = (h_12h + l_12h + c_12h) / 3.0
    hl_range = h_12h - l_12h
    
    # R3 and S3 levels (Camarilla formula)
    r3_12h = c_12h + (hl_range * 1.1 / 4.0)
    s3_12h = c_12h - (hl_range * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h timeframe (use previous 12h bar's levels)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter (more stable)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r3_val = r3_12h_aligned[i]
        s3_val = s3_12h_aligned[i]
        
        # Long logic: price breaks above R3 with volume spike and 1d uptrend
        long_condition = (close_val > r3_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below S3 with volume spike and 1d downtrend
        short_condition = (close_val < s3_val) and volume_spike[i] and (close_val < ema_val)
        
        # Exit logic: trend reversal
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        # Minimum holding period: 2 bars (24h for 12h timeframe)
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0