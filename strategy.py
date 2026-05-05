#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above weekly Donchian upper channel (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below weekly Donchian lower channel (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above weekly Donchian middle (mean of upper/lower) OR ADX drops below 20
# Uses discrete sizing 0.30 to balance return and risk
# Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe
# Weekly Donchian provides robust structure from higher timeframe
# 1d ADX > 25 ensures we only trade strong trends, avoiding whipsaws in ranging markets
# Volume confirmation validates breakout strength
# Works in bull markets (breakouts with strong uptrend) and bear markets (breakdowns with strong downtrend)

name = "1d_WeeklyDonchian20_Breakout_1dADX25_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    upper_channel = high_20
    lower_channel = low_20
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Align weekly Donchian channels to 1d timeframe (wait for completed weekly bar)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    middle_channel_aligned = align_htf_to_ltf(prices, df_1w, middle_channel)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX (14+)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 1d timeframe (no additional delay needed)
    adx_aligned = adx  # Already on 1d timeframe
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(middle_channel_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper channel, ADX > 25, volume confirmation, in session
            if close[i] > upper_channel_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below weekly Donchian lower channel, ADX > 25, volume confirmation, in session
            elif close[i] < lower_channel_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian middle channel OR ADX drops below 20
            if close[i] < middle_channel_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian middle channel OR ADX drops below 20
            if close[i] > middle_channel_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals