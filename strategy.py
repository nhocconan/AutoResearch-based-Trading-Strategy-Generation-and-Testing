#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX(25) trend filter and volume spike
# Uses 1d ADX for trend strength, 6h Donchian for breakout signals, volume spike (>2.0x 20-bar avg) for confirmation
# Exit: time-based exit after 3 bars (18 hours) to avoid whipsaw in ranging markets
# Works in both bull/bear: breakouts capture momentum, ADX filter avoids weak trends, time exit limits drawdown
# Target: 60-120 total trades over 4 years (15-30/year) to stay within fee limits

name = "6h_Donchian20_1dADX25_VolumeSpike_v1"
timeframe = "6h"
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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(25) trend filter
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
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
    
    atr_1d = wilder_smooth(tr, 25)
    dm_plus_smooth = wilder_smooth(dm_plus, 25)
    dm_minus_smooth = wilder_smooth(dm_minus, 25)
    
    # DI+ = 100 * smoothed +DM / ATR, DI- = 100 * smoothed -DM / ATR
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX = smoothed DX
    adx_1d = wilder_smooth(dx, 25)
    
    # Calculate 6h Donchian channels (20-period)
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper_6h, donchian_lower_6h = donchian_channels(high, low, 20)
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe (primary)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_upper_6h[i]) or 
            np.isnan(donchian_lower_6h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > Upper Donchian AND strong trend (ADX > 25) AND volume spike
            if close[i] > donchian_upper_6h[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short breakdown: price < Lower Donchian AND strong trend (ADX > 25) AND volume spike
            elif close[i] < donchian_lower_6h[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # Exit: time-based exit after 3 bars (18 hours) or reverse signal
            if bars_since_entry >= 3:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif close[i] < donchian_lower_6h[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                # Reverse to short
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            bars_since_entry += 1
            # Exit: time-based exit after 3 bars (18 hours) or reverse signal
            if bars_since_entry >= 3:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif close[i] > donchian_upper_6h[i] and adx_1d_aligned[i] > 25 and volume_filter[i]:
                # Reverse to long
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals