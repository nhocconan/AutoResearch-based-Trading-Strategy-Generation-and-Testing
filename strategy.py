#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with 1d ADX trend filter and volume spike confirmation
# Long when price breaks above weekly Donchian upper (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20) on 1d
# Short when price breaks below weekly Donchian lower (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price crosses weekly Donchian midpoint OR ADX < 20 (trend weakens)
# Uses discrete sizing 0.30 to balance return and risk
# Target: 30-80 total trades over 4 years (7-20/year) for 1d timeframe
# Weekly Donchian provides robust structure from higher timeframe
# 1d ADX filters for trending markets to avoid whipsaws in ranging conditions
# Volume confirmation ensures breakout authenticity
# Works in bull markets (breakouts with strong uptrend) and bear markets (breakdowns with strong downtrend)

name = "1d_Donchian20_1dADX_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channel (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Mid = (Upper + Lower)/2
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_upper = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1w_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align weekly Donchian levels to 1d timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX(14)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    high_low = high_1d - low_1d
    high_close_prev = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    low_close_prev = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    
    # +DM = max(high - high_prev, 0) if > max(low_prev - low, 0) else 0
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # DI+ = 100 * smoothed +DM / smoothed TR
    # DI- = 100 * smoothed -DM / smoothed TR
    # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
    # ADX = smoothed DX
    di_plus = np.where(tr_smooth > 0, 100 * plus_dm_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth > 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 1d timeframe (no additional delay needed for ADX)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
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
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper, ADX > 25, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below weekly Donchian lower, ADX > 25, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian mid OR ADX < 20 (trend weakens)
            if close[i] < donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian mid OR ADX < 20 (trend weakens)
            if close[i] > donchian_mid_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals