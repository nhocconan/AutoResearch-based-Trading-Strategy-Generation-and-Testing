#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses Camarilla R3/S3 levels from daily pivots for breakout entries, filtered by weekly trend (EMA34) and volume spikes.
Enters long when price breaks above R3 AND weekly close > EMA34 AND volume > 2.0 * 20-period average.
Enters short when price breaks below S3 AND weekly close < EMA34 AND volume > 2.0 * 20-period average.
Exits when price returns to the opposite Camarilla level (R3 for shorts, S3 for longs) or weekly trend reverses.
Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with tight entry conditions.
Uses proven Camarilla structure that works in both bull and bear markets by following weekly trend while using daily pivots for precise swing-point entries.
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
    
    # Get 1d data for Camarilla pivots and weekly for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # But standard Camarilla uses: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # We'll use the more common multiplier of 1.1 for R3/S3
    cam_multiplier = 1.1
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r3 = close_1d + cam_multiplier * (high_1d - low_1d)
    camarilla_s3 = close_1d - cam_multiplier * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Weekly EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need daily Camarilla (requires 1d data), weekly EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        ema_val = ema_34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R3/S3 with weekly trend filter AND volume
            # Long: price breaks above R3 AND weekly uptrend AND volume confirmation
            long_condition = (close_val > r3_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S3 AND weekly downtrend AND volume confirmation
            short_condition = (close_val < s3_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to S3 level OR weekly trend breaks
            exit_condition = (close_val <= s3_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to R3 level OR weekly trend breaks
            exit_condition = (close_val >= r3_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0