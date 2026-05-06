#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation on 1h
# Uses 4h Camarilla pivot levels (R3/S3) for directional bias
# Requires price to break above R3 with volume > 1.5x 20-period average for long
# Requires price to break below S3 with volume > 1.5x 20-period average for short
# Uses 1h ADX(25) to filter for trending markets only
# Designed for 1h timeframe to target 60-150 total trades over 4 years (15-37/year)
# Works in both bull/bear: captures breakouts in trending markets, avoids false signals in ranges

name = "4h_Camarilla_R3S3_Breakout_1hADX25_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for 4h
    # Pivot = (H + L + C) / 3
    # R3 = H + 1.1 * (H - L)
    # S3 = L - 1.1 * (H - L)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r3_4h = high_4h + 1.1 * (high_4h - low_4h)
    s3_4h = low_4h - 1.1 * (high_4h - low_4h)
    
    # Calculate 1h ADX(25) trend filter
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = np.abs(high_1h[1:] - low_1h[1:])
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where((high_1h[1:] - high_1h[:-1]) > (low_1h[:-1] - low_1h[1:]), 
                       np.maximum(high_1h[1:] - high_1h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where((low_1h[:-1] - low_1h[1:]) > (high_1h[1:] - high_1h[:-1]), 
                        np.maximum(low_1h[:-1] - low_1h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr_1h = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    # DI+ = 100 * smoothed +DM / ATR, DI- = 100 * smoothed -DM / ATR
    di_plus = np.where(atr_1h != 0, 100 * dm_plus_smooth / atr_1h, 0)
    di_minus = np.where(atr_1h != 0, 100 * dm_minus_smooth / atr_1h, 0)
    
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX = smoothed DX
    adx_1h = wilder_smooth(dx, 25)
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(adx_1h_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume confirmation in trending market
            if (close[i] > r3_4h_aligned[i] and 
                adx_1h_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 with volume confirmation in trending market
            elif (close[i] < s3_4h_aligned[i] and 
                  adx_1h_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below R3
            if close[i] < r3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above S3
            if close[i] > s3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals