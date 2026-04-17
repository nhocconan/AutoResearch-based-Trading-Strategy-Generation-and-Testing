#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal + 1d Volume Spike + ADX Trend Filter
Long: Price touches S3/S4 with rejection + volume spike + ADX > 25
Short: Price touches R3/R4 with rejection + volume spike + ADX > 25
Exit: Opposite touch or ADX < 20
Uses Camarilla levels from daily for institutional reversal zones.
ADX filter ensures we only trade in trending markets to avoid whipsaws.
Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    diff_1d = high_1d - low_1d
    r4_1d = close_1d + 1.5 * diff_1d
    r3_1d = close_1d + 1.1 * diff_1d
    s3_1d = close_1d - 1.1 * diff_1d
    s4_1d = close_1d - 1.5 * diff_1d
    
    # Align Camarilla levels to 12h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate ADX for trend strength (14-period)
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * smoothed(+DM) / ATR
    # -DI = 100 * smoothed(-DM) / ATR
    # ADX = smoothed(|+DI - -DI| / (+DI + -DI))
    
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    high_low = high - low
    high_prev_close = np.abs(high - prev_close)
    low_prev_close = np.abs(low - prev_close)
    tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    
    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
                else:
                    result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Volume spike: current volume > 2x 20-period volume SMA
    vol_sma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_sma[:19] = np.nan
    vol_sma[-1:] = np.nan
    volume_spike = volume > 2.0 * vol_sma
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 40)  # need enough data for ADX and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx[i]
        r4 = r4_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        if position == 0:
            # Long: Price touches S3/S4 with rejection + volume spike + ADX > 25
            # Touch defined as low touching or penetrating the level
            touch_s3 = low[i] <= s3
            touch_s4 = low[i] <= s4
            # Rejection: close back above the level (shows buying pressure)
            reject_s3 = touch_s3 and close[i] > s3
            reject_s4 = touch_s4 and close[i] > s4
            
            if (reject_s3 or reject_s4) and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price touches R3/R4 with rejection + volume spike + ADX > 25
            elif (high[i] >= r3 or high[i] >= r4) and vol_spike and adx_val > 25:
                # Rejection: close back below the level (shows selling pressure)
                reject_r3 = high[i] >= r3 and close[i] < r3
                reject_r4 = high[i] >= r4 and close[i] < r4
                if reject_r3 or reject_r4:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Price touches R3/R4 or ADX < 20 (trend weakening)
            if (high[i] >= r3 or high[i] >= r4) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price touches S3/S4 or ADX < 20 (trend weakening)
            if (low[i] <= s3 or low[i] <= s4) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_Volume_ADX"
timeframe = "12h"
leverage = 1.0