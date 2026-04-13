#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
    # Long when price breaks above 20-day high + 1w close > 1w open (bullish weekly) + volume > 1.5x 20-day avg
    # Short when price breaks below 20-day low + 1w close < 1w open (bearish weekly) + volume > 1.5x 20-day avg
    # Exit when price returns to 10-day midpoint or opposite breakout
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 30-100 total trades over 4 years (~7-25/year)
    # Donchian provides clear structure; weekly trend filters counter-trend whipsaws; volume confirms strength
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) with min_periods
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-day high and low
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 10-day midpoint for exit
    high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    midpoint_10 = (high_10 + low_10) / 2
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're on 1d timeframe)
    # But we still use align_htf_to_ltf for consistency and proper handling
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    midpoint_10_aligned = align_htf_to_ltf(prices, df_1d, midpoint_10)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1w trend: bullish if weekly close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    
    # Align 1w trend to 1d timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(midpoint_10_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        volume_confirm = volume_1d[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        bullish_breakout = close[i] > high_20_aligned[i] and weekly_bullish_aligned[i] > 0.5 and volume_confirm
        bearish_breakout = close[i] < low_20_aligned[i] and weekly_bearish_aligned[i] > 0.5 and volume_confirm
        
        # Exit conditions: price returns to 10-day midpoint or opposite breakout
        long_exit = close[i] < midpoint_10_aligned[i] or bearish_breakout
        short_exit = close[i] > midpoint_10_aligned[i] or bullish_breakout
        
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

name = "1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0