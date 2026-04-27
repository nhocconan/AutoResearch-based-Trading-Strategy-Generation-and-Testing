#!/usr/bin/env python3
"""
12h Camarilla Pivot R3/S3 Breakout with 1d Volume Spike and ADX Trend Filter
Long: Close breaks above R3 with volume > 1.5x daily average and ADX > 25
Short: Close breaks below S3 with volume > 1.5x daily average and ADX > 25
Exit: Close crosses below/above daily VWAP or ADX drops below 20
Position size: 0.25
Target: 15-25 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivots, volume, VWAP, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_num = (typical_price * df_1d['volume']).cumsum()
    vwap_den = df_1d['volume'].cumsum()
    vwap = vwap_num / vwap_den
    vwap_array = vwap.values
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_array)
    
    # Calculate daily average volume (20-period)
    vol_20 = df_1d['volume'].rolling(20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    
    # Calculate Camarilla pivots from previous day
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    # We use previous day's values (shifted by 1)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate ADX (14-period) on daily data
    # ADX requires +DI, -DI, and DX
    high_diff = df_1d['high'].diff()
    low_diff = -df_1d['low'].diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(arr)):
            if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
            else:
                result[i] = np.nan
        return result
    
    tr_smooth = wilder_smooth(tr.values, 14)
    plus_di_smooth = wilder_smooth(plus_dm, 14)
    minus_di_smooth = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    dx = np.full_like(tr_smooth, np.nan)
    mask = tr_smooth != 0
    dx[mask] = 100 * np.abs(plus_di_smooth[mask] - minus_di_smooth[mask]) / tr_smooth[mask]
    
    adx = wilder_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Start after we have enough data for all indicators
    start_idx = 50  # Need 20 for vol, 14+14 for ADX, plus buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(vol_avg_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        vol_now = volume[i]
        vwap_val = vwap_aligned[i]
        vol_avg_val = vol_avg_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        vol_filter = vol_now > 1.5 * vol_avg_val if vol_avg_val > 0 else False
        
        # ADX trend filter: ADX > 25 for strong trend
        strong_trend = adx_val > 25
        
        # ADX exit filter: ADX < 20 for weak trend
        weak_trend = adx_val < 20
        
        if position == 0:
            # Look for long: price breaks above R3 with volume and trend
            if price_now > r3_val and vol_filter and strong_trend:
                signals[i] = size
                position = 1
            # Look for short: price breaks below S3 with volume and trend
            elif price_now < s3_val and vol_filter and strong_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP or trend weakens
            if price_now < vwap_val or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above VWAP or trend weakens
            if price_now > vwap_val or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Volume_ADX"
timeframe = "12h"
leverage = 1.0