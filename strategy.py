#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with 1d volume spike and ADX regime filter
# Long when price breaks above 1w Donchian upper channel AND 1d volume > 1.5 * 20-period avg volume AND 1d ADX > 25
# Short when price breaks below 1w Donchian lower channel AND 1d volume > 1.5 * 20-period avg volume AND 1d ADX > 25
# Exit when price crosses back inside the 1w Donchian channel OR volume drops below average
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe
# 1w Donchian provides robust multi-week structure
# Volume spike confirms breakout conviction
# ADX > 25 ensures we only trade in trending regimes, avoiding choppy whipsaws
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "1d_Donchian20_1w_VolumeSpike_ADX"
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
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least one completed 1w bar for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian(20) channels (based on previous 20 completed 1w bars)
    # Upper = max(high_1w over last 20 periods), Lower = min(low_1w over last 20 periods)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_upper = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_1w_series.rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian channels to 1d timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Get 1d data ONCE before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume and ADX calculations
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 1.5 * 20-period average volume
    volume_1d_series = pd.Series(volume_1d)
    avg_volume_20 = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.5 * avg_volume_20)
    
    # Calculate 1d ADX(14) for regime filter
    # ADX requires +DI, -DI, and TR calculations
    # True Range (TR) = max(high-low, abs(high-previous_close), abs(low-previous_close))
    prev_close = np.append([close_1d[0]], close_1d[:-1])
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close)
    tr3 = np.abs(low_1d - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI calculations
    up_move = high_1d - np.append([high_1d[0]], high_1d[:-1])
    down_move = np.append([low_1d[0]], low_1d[:-1]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 1d timeframe (no additional delay needed as they're already 1d)
    volume_confirm_aligned = volume_confirm  # already 1d
    adx_aligned = adx  # already 1d
    
    # Session filter: 00-24 UTC (always in session for 1d)
    in_session = np.ones(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(adx_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 1w Donchian upper, volume confirmation, ADX > 25 (trending)
            if close[i] > donchian_upper_aligned[i] and volume_confirm_aligned[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1w Donchian lower, volume confirmation, ADX > 25 (trending)
            elif close[i] < donchian_lower_aligned[i] and volume_confirm_aligned[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses back inside 1w Donchian channel OR volume drops below average
            if close[i] < donchian_upper_aligned[i] and close[i] > donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif not volume_confirm_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses back inside 1w Donchian channel OR volume drops below average
            if close[i] < donchian_upper_aligned[i] and close[i] > donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif not volume_confirm_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals