#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25
# Exit when price crosses Camarilla pivot point (PP) OR ADX < 20 (trend weakening)
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-25 trades/year per symbol.
# Camarilla levels provide precise intraday support/resistance, volume confirms participation,
# ADX ensures we only trade in trending conditions to avoid chop whipsaws.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_Camarilla_R3S3_VolumeSpike_1dADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h data (using previous bar's OHLC)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r3_12h = close_12h + (high_12h - low_12h) * 1.1 / 2.0
    s3_12h = close_12h - (high_12h - low_12h) * 1.1 / 2.0
    
    # Align Camarilla levels to prices timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume confirmation: volume > 2.0x 20-period average
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Align volume filter to prices timeframe
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d.astype(float))
    
    # Calculate 1d ADX for trend filter (ADX > 25 = strong trend)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) if not np.all(np.isnan(data[1:period])) else np.nan
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_period = 14
    tr_smooth = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, dm_plus_smooth / tr_smooth * 100, 0)
    di_minus = np.where(tr_smooth != 0, dm_minus_smooth / tr_smooth * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    # ADX filter: > 25 = strong trend
    adx_filter = adx > 25
    
    # Align ADX filter to prices timeframe
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or 
            np.isnan(adx_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND volume confirmation AND ADX > 25
            if (close[i] > r3_aligned[i] and 
                volume_filter_aligned[i] > 0.5 and 
                adx_filter_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND volume confirmation AND ADX > 25
            elif (close[i] < s3_aligned[i] and 
                  volume_filter_aligned[i] > 0.5 and 
                  adx_filter_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below PP OR ADX < 20 (trend weakening)
            if (close[i] < pp_aligned[i] or 
                adx_filter_aligned[i] < 0.5):  # ADX <= 25 (we used >25 for entry, so <=0.5 means false)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above PP OR ADX < 20 (trend weakening)
            if (close[i] > pp_aligned[i] or 
                adx_filter_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals