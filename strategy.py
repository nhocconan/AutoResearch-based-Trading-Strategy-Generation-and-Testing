#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and chop regime filter
    # Long when price breaks above Donchian upper (20-period high) + 12h volume > 1.3x average + chop < 61.8 (trending)
    # Short when price breaks below Donchian lower (20-period low) + 12h volume > 1.3x average + chop < 61.8 (trending)
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 75-200 total trades over 4 years (~19-50/year)
    # Donchian provides clear structure; volume confirms breakout strength; chop filter avoids false breakouts in ranging markets
    # Works in both bull and bear: volume + chop filter ensures we only trade strong trending moves
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for volume and chop (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) - using 4h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period) with min_periods
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate 12h Choppiness Index (14-period) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = np.abs(np.roll(high_12h, 1) - low_12h)  # |high_prev - low_curr|
    tr2 = np.abs(np.roll(low_12h, 1) - high_12h)  # |low_prev - high_curr|
    tr3 = np.abs(high_12h - low_12h)              # |high_curr - low_curr|
    tr1[0] = np.nan  # First value invalid due to roll
    tr2[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR) / (log10(n) * (highest_high - lowest_low))) / log10(n)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    denominator = lowest_low - highest_high
    chop = np.where(
        (denominator != 0) & (~np.isnan(denominator)) & (highest_high != lowest_low) & (~np.isnan(atr_sum)),
        100 * np.log10(atr_sum / (np.log10(14) * np.abs(denominator))) / np.log10(14),
        50  # Default to neutral when invalid
    )
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume confirmation: current 12h volume > 1.3 * 20-period average
    vol_12h_current = df_12h['volume'].values
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_current)
    volume_confirm = vol_12h_aligned > 1.3 * vol_ma_aligned
    
    # Chop regime filter: trending market (chop < 61.8)
    trending_regime = chop_aligned < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper[i] and volume_confirm[i] and trending_regime[i]
        bearish_breakout = close[i] < donchian_lower[i] and volume_confirm[i] and trending_regime[i]
        
        # Exit conditions: price returns to middle of channel or opposite breakout
        donchian_middle = (donchian_upper + donchian_lower) / 2
        long_exit = close[i] < donchian_middle[i] or bearish_breakout
        short_exit = close[i] > donchian_middle[i] or bullish_breakout
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0