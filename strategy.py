#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1h volume spike and 4h ADX trend filter
# Uses 1h HTF for volume confirmation to reduce noise and false breakouts
# Uses 4h ADX > 25 to ensure we only trade in trending markets, avoiding chop
# Camarilla levels from 1d provide institutional support/resistance
# Designed for 4h timeframe targeting 20-40 trades/year with discrete sizing (0.25)
# The combination of volume spike + ADX filter should significantly reduce overtrading
# while maintaining edge in both bull and bear markets through proper trend alignment

name = "4h_Camarilla_R3S3_Breakout_1hVolumeSpike_4hADX25_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1h data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 4h ADX for trend filter (period=14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original indices
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            first_avg = np.nansum(values[1:period+1])
            result[period] = first_avg
            for i in range(period+1, len(values)):
                result[i] = result[i-1] - (result[i-1]/period) + values[i]
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate camarilla levels: R3, S3 from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 2.0x 20-period EMA on 1h volume (more responsive than 4h)
    vol_1h = df_1h['volume'].values
    vol_series = pd.Series(vol_1h)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 2.0 x 20-period EMA (aligned to 4h)
        # Get the corresponding 1h volume for this 4h bar
        volume_confirmed = False
        if i >= 4:  # Need at least 4 1h bars in each 4h bar
            # Approximate: use current volume if we're at a 4h boundary, else use recent 1h volume
            # Simplified: check if current volume is elevated (works as proxy)
            volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: close breaks above R3 + volume confirmation + ADX > 25 (uptrend)
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                adx_4h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S3 + volume confirmation + ADX > 25 (downtrend)
            elif (close[i] < s3_aligned[i] and volume_confirmed and 
                  adx_4h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] < s3_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 (mean reversion) OR ADX < 20 (trend weakening)
            if close[i] > r3_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals