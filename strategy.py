#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Slow_Plus
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Enhancements: Added ADX(14) trend strength filter to avoid whipsaws in sideways markets.
# Uses higher volume threshold (2.0x average) and only enters when price is near 1d VWAP.
# Designed to generate ~15-25 trades/year on 4h to avoid fee drag while maintaining edge.
# Long when 1d trend up (close > EMA34), ADX > 25, price breaks above R3, volume > 2x average, and close > 1d VWAP.
# Short when 1d trend down (close < EMA34), ADX > 25, price breaks below S3, volume > 2x average, and close < 1d VWAP.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Slow_Plus"
timeframe = "4h"
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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Align 1d EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    # Calculate smoothed +DM, -DM, TR
    smoothed_plus_dm = wilders_smoothing(plus_dm, 14)
    smoothed_minus_dm = wilders_smoothing(minus_dm, 14)
    smoothed_tr = wilders_smoothing(tr, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.full_like(close_1d, np.nan)
    di_minus = np.full_like(close_1d, np.nan)
    valid = (~np.isnan(smoothed_plus_dm)) & (~np.isnan(smoothed_minus_dm)) & (smoothed_tr != 0)
    di_plus[valid] = (smoothed_plus_dm[valid] / smoothed_tr[valid]) * 100
    di_minus[valid] = (smoothed_minus_dm[valid] / smoothed_tr[valid]) * 100
    
    # Calculate DX and ADX
    dx = np.full_like(close_1d, np.nan)
    dx_valid = (~np.isnan(di_plus)) & (~np.isnan(di_minus)) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])) * 100
    
    adx_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 27:  # Need 14 for DX + 14 for ADX smoothing
        # First ADX is average of first 14 DX values
        first_adx_idx = 27  # 14 (for DX) + 13 (for smoothing to get first smoothed value)
        if first_adx_idx < len(dx):
            valid_dx = dx[14:first_adx_idx+1]  # DX values from index 14 to 27
            if len(valid_dx) > 0:
                adx_1d[first_adx_idx] = np.nanmean(valid_dx)
                # Subsequent ADX values: smoothed = prev * (13/14) + current * (1/14)
                for i in range(first_adx_idx + 1, len(dx)):
                    if not np.isnan(dx[i]):
                        adx_1d[i] = adx_1d[i-1] * (13/14) + dx[i] * (1/14)
    
    # Align 1d EMA34 and ADX to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Camarilla levels for each 1d bar: R3, S3
    camarilla_r3_1d = np.full_like(close_1d, np.nan)
    camarilla_s3_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_r3_1d[i] = close_1d[i] + 1.1 * (high_1d[i] - low_1d[i]) / 2
            camarilla_s3_1d[i] = close_1d[i] - 1.1 * (high_1d[i] - low_1d[i]) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate 1d VWAP for additional filter
    vwap_1d = np.full_like(close_1d, np.nan)
    cumulative_volume = np.full_like(close_1d, np.nan)
    cumulative_price_volume = np.full_like(close_1d, np.nan)
    
    for i in range(len(df_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]) or np.isnan(volume_1d[i]):
            if i > 0:
                vwap_1d[i] = vwap_1d[i-1]
                cumulative_volume[i] = cumulative_volume[i-1]
                cumulative_price_volume[i] = cumulative_price_volume[i-1]
            continue
            
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        price_volume = typical_price * volume_1d[i]
        
        if i == 0:
            cumulative_volume[i] = volume_1d[i]
            cumulative_price_volume[i] = price_volume
        else:
            cumulative_volume[i] = cumulative_volume[i-1] + volume_1d[i]
            cumulative_price_volume[i] = cumulative_price_volume[i-1] + price_volume
            
        if cumulative_volume[i] != 0:
            vwap_1d[i] = cumulative_price_volume[i] / cumulative_volume[i]
        else:
            vwap_1d[i] = vwap_1d[i-1] if i > 0 else typical_price
    
    # Align 1d VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(volume_ratio[i]) or np.isnan(vwap_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend, trend strength, and price relative to VWAP
        trend_up = close[i] > ema34_1d_aligned[i]
        strong_trend = adx_1d_aligned[i] > 25
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        
        if position == 0:
            # Enter long: 1d trend up + strong trend + price breaks above R3 + volume confirmation + price above VWAP
            if trend_up and strong_trend and close[i] > camarilla_r3_1d_aligned[i] and volume_ratio[i] > 2.0 and price_above_vwap:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d trend down + strong trend + price breaks below S3 + volume confirmation + price below VWAP
            elif not trend_up and strong_trend and close[i] < camarilla_s3_1d_aligned[i] and volume_ratio[i] > 2.0 and price_below_vwap:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d trend turns down or weak trend or price breaks below S3 or price falls below VWAP
            if not trend_up or not strong_trend or close[i] < camarilla_s3_1d_aligned[i] or not price_above_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d trend turns up or weak trend or price breaks above R3 or price rises above VWAP
            if trend_up or not strong_trend or close[i] > camarilla_r3_1d_aligned[i] or not price_below_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals