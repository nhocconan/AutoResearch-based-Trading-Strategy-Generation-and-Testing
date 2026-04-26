#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: Trade Camarilla R3/S3 breakouts in the direction of 1d EMA34 trend with volume spike confirmation.
Long when: price breaks above Camarilla R3 + 1d EMA34 up + volume > 1.5x avg volume.
Short when: price breaks below Camarilla S3 + 1d EMA34 down + volume > 1.5x avg volume.
Exit when: price reverts to Camarilla midpoint (R3/S3 avg) or 1d EMA34 flips.
Uses 0.25 position size to limit fee drag. Targets ~25-35 trades/year.
Works in bull/bear: trend filter avoids counter-trend trades, volume confirms conviction.
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
    
    # Camarilla levels from previous day (need OHLC)
    # We'll calculate from daily data using mtf_data
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for each day: based on previous day's range
    # R3 = close + 1.1*(high-low)*1.1/4? Actually: R3 = close + 1.1*(high-low)
    # Standard Camarilla: 
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    # We'll use R3 and S3
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day has no previous, fill with NaN
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_r3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_s3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d)
    camarilla_mid_1d = (camarilla_r3_1d + camarilla_s3_1d) / 2  # midpoint
    
    # Align to 4h
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_mid_4h = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # 1d EMA34 for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # EMA34 slope (rising/falling)
    ema_slope = np.diff(ema_34_4h, prepend=ema_34_4h[0])
    ema_up = ema_slope > 0
    ema_down = ema_slope < 0
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for vol MA, 34 for EMA, 1 for Camarilla (uses prev day)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or
            np.isnan(camarilla_mid_4h[i]) or np.isnan(ema_34_4h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_ok = vol_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for breakout in trend direction
            long_break = (close_val > camarilla_r3_4h[i]) and ema_up[i] and vol_ok
            short_break = (close_val < camarilla_s3_4h[i]) and ema_down[i] and vol_ok
            
            if long_break:
                signals[i] = size
                position = 1
            elif short_break:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or trend flips
            if close_val < camarilla_mid_4h[i] or not ema_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or trend flips
            if close_val > camarilla_mid_4h[i] or not ema_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0