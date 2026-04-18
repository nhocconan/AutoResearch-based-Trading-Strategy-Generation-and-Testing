#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with weekly ADX trend filter and volume confirmation.
# Camarilla levels provide precise support/resistance levels derived from prior day's range.
# Weekly ADX filter ensures we only trade in trending markets to avoid whipsaws in ranges.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above resistance) and bear markets (breakouts below support).
name = "12h_Camarilla_R1_S1_Breakout_WeeklyADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ADX filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for Camarilla pivots (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly ADX (14-period)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range calculation for weekly
    tr1_w = high_w[1:] - low_w[1:]
    tr2_w = np.abs(high_w[1:] - close_w[:-1])
    tr3_w = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    
    # Directional Movement
    dm_plus_w = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                         np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus_w = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                          np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus_w = np.concatenate([[0], dm_plus_w])
    dm_minus_w = np.concatenate([[0], dm_minus_w])
    
    # Smoothed values using Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_w = wilders_smooth(tr_w, 14)
    dm_plus_smooth = wilders_smooth(dm_plus_w, 14)
    dm_minus_smooth = wilders_smooth(dm_minus_w, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_w > 0, 100 * dm_plus_smooth / atr_w, 0)
    di_minus = np.where(atr_w > 0, 100 * dm_minus_smooth / atr_w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align weekly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate daily Camarilla pivot levels
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_d + low_d + close_d) / 3
    
    # Camarilla levels
    range_d = high_d - low_d
    r1 = close_d + (range_d * 1.1 / 12)
    s1 = close_d - (range_d * 1.1 / 12)
    r2 = close_d + (range_d * 1.1 / 6)
    s2 = close_d - (range_d * 1.1 / 6)
    r3 = close_d + (range_d * 1.1 / 4)
    s3 = close_d - (range_d * 1.1 / 4)
    r4 = close_d + (range_d * 1.1 / 2)
    s4 = close_d - (range_d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 24-period average volume for confirmation (2 days of 12h data)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_24[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND trend filter
            long_breakout = close[i] > r1_aligned[i]
            if vol_confirm and trend_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] < s1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR ADX drops below 20 (trend weakening)
            exit_condition = close[i] > r1_aligned[i] or adx_aligned[i] < 20
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals