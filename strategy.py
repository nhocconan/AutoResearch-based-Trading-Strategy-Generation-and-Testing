#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with volume confirmation and ADX trend filter
# Long when price breaks above 12h Donchian upper channel AND volume > 1.5 * avg_volume(20) AND ADX(14) > 25
# Short when price breaks below 12h Donchian lower channel AND volume > 1.5 * avg_volume(20) AND ADX(14) > 25
# Exit when price crosses back to 12h Donchian middle channel OR ADX < 20 (trend weakening)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# 12h Donchian provides robust structure from higher timeframe
# Volume confirmation reduces false breakouts
# ADX filter ensures trades only in trending markets, avoiding whipsaws in ranging conditions
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_Donchian12h_UpperLower_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (Upper + Lower) / 2
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donchian_20_upper = high_12h_series.rolling(window=20, min_periods=20).max().values
    donchian_20_lower = low_12h_series.rolling(window=20, min_periods=20).min().values
    donchian_20_middle = (donchian_20_upper + donchian_20_lower) / 2.0
    
    # Align 12h Donchian levels to 4h timeframe (wait for completed 12h bar)
    donchian_20_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_upper)
    donchian_20_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_lower)
    donchian_20_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_middle)
    
    # Calculate ADX(14) on 4h for trend filter
    # ADX requires +DM, -DM, TR, then smoothed averages
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr.values
    minus_di = 100 * minus_dm_smooth / atr.values
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_20_upper_aligned[i]) or np.isnan(donchian_20_lower_aligned[i]) or 
            np.isnan(donchian_20_middle_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian upper, volume confirmation, ADX > 25, in session
            if close[i] > donchian_20_upper_aligned[i] and volume_confirm[i] and adx[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian lower, volume confirmation, ADX > 25, in session
            elif close[i] < donchian_20_lower_aligned[i] and volume_confirm[i] and adx[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 12h Donchian middle OR ADX < 20 (trend weakening)
            if close[i] < donchian_20_middle_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 12h Donchian middle OR ADX < 20 (trend weakening)
            if close[i] > donchian_20_middle_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals