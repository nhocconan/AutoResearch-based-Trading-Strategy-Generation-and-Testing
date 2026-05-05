#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with 1d ADX trend filter and volume spike confirmation
# Long when price breaks above weekly Donchian upper channel (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20) on 1d
# Short when price breaks below weekly Donchian lower channel (20) AND 1d ADX > 25 AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price crosses back below/above weekly Donchian middle OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe
# Weekly Donchian provides robust structure from higher timeframe
# 1d ADX > 25 ensures we only trade strong trends, avoiding whipsaws in ranging markets
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (breakouts with strong uptrend) and bear markets (breakdowns with strong downtrend)

name = "1d_WeeklyDonchian20_Breakout_1dADX25_VolumeSpike"
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
    if len(df_1w) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (based on previous 20 weekly bars)
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (Upper + Lower)/2
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_upper = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1w_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align weekly Donchian channels to 1d timeframe (wait for completed weekly bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle)
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align 1d ADX to 1d timeframe (no additional delay needed)
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
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper, ADX > 25, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower, ADX > 25, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Donchian middle OR volume drops below average
            if close[i] < donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly Donchian middle OR volume drops below average
            if close[i] > donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals