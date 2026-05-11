#!/usr/bin/env python3
name = "1d_1W_Camarilla_R3S3_Breakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    high_prev = df_1w['high'].values
    low_prev = df_1w['low'].values
    close_prev = df_1w['close'].values
    
    # Shift to use previous week's data (no look-ahead)
    high_prev = np.roll(high_prev, 1)
    low_prev = np.roll(low_prev, 1)
    close_prev = np.roll(close_prev, 1)
    high_prev[0] = high_prev[1] if len(high_prev) > 1 else high_prev[0]
    low_prev[0] = low_prev[1] if len(low_prev) > 1 else low_prev[0]
    close_prev[0] = close_prev[1] if len(close_prev) > 1 else close_prev[0]
    
    # Camarilla calculation
    range_prev = high_prev - low_prev
    camarilla_r3 = close_prev + 1.1 * range_prev * 1.1000  # R3
    camarilla_s3 = close_prev - 1.1 * range_prev * 1.1000  # S3
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to daily
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R3 with volume and above weekly EMA34 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_surge and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume and below weekly EMA34 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level (R3/S3) or weekly EMA
            if position == 1:
                # Exit long: price returns to S3 or below EMA34
                if (close[i] < camarilla_s3_aligned[i]) or (close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R3 or above EMA34
                if (close[i] > camarilla_r3_aligned[i]) or (close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals