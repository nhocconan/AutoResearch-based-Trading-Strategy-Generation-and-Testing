#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Donchian(20) breakout from 6h timeframe with 12h ADX trend filter and volume spike confirmation
# Long when price breaks above 6h Donchian upper band AND 12h ADX > 25 AND volume > 2.0 * 20-period avg volume on 6h
# Short when price breaks below 6h Donchian lower band AND 12h ADX > 25 AND volume > 2.0 * 20-period avg volume on 6h
# Exit when price crosses back below/above 6h Donchian middle band OR ADX drops below 20
# Uses discrete sizing 0.25 to balance return and risk
# Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe
# Donchian provides robust price channel structure
# 12h ADX filters for trending markets only to avoid whipsaws in ranging markets
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (breakouts with strong uptrend) and bear markets (breakdowns with strong downtrend)

name = "6h_Donchian20_12hADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need at least one completed 6h bar for Donchian
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band = highest high over past 20 periods
    # Lower band = lowest low over past 20 periods
    # Middle band = (upper + lower) / 2
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    donchian_upper = high_6h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_6h_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align 6h Donchian levels to 6h timeframe (wait for completed 6h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_6h, donchian_middle)
    
    # Get 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX (14-period)
    # ADX requires +DI, -DI, and TR calculations
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    close_12h_series = pd.Series(close_12h)
    
    # True Range
    tr1 = high_12h_series - low_12h_series
    tr2 = abs(high_12h_series - close_12h_series.shift(1))
    tr3 = abs(low_12h_series - close_12h_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_12h_series - high_12h_series.shift(1)
    down_move = low_12h_series.shift(1) - low_12h_series
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth DM and TR
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    
    # Calculate ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align 12h ADX to 6h timeframe (wait for completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
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
            # Long: Price breaks above 6h Donchian upper band, ADX > 25, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 6h Donchian lower band, ADX > 25, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 6h Donchian middle band OR ADX drops below 20
            if close[i] < donchian_middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 6h Donchian middle band OR ADX drops below 20
            if close[i] > donchian_middle_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals