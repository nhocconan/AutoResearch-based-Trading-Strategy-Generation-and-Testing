#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
- Long when price breaks above 20-day high AND weekly close > weekly EMA50 (bullish regime)
- Short when price breaks below 20-day low AND weekly close < weekly EMA50 (bearish regime)
- Volume confirmation: current volume > 2.0 * 20-day average volume
- Exit on opposite Donchian breakout (20-day low for long exit, 20-day high for short exit)
- Uses 1d primary with 1w HTF to target 30-100 trades over 4 years (7-25/year)
- Donchian channels provide clear trend-following structure; EMA50 filters regime; volume confirms momentum
- Designed to work in both bull (breakouts with trend) and bear (breakouts against trend filtered out) markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels on daily data
    # Highest high of last 20 days, lowest low of last 20 days
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Donchian upper (20-day high) and lower (20-day low)
    donchian_upper = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned since we're using 1d data)
    # But we need to shift by 1 to avoid look-ahead (use previous day's levels for today's breakout)
    donchian_upper_aligned = np.roll(donchian_upper, 1)
    donchian_lower_aligned = np.roll(donchian_lower, 1)
    # Set first value to NaN since we don't have previous day's levels
    donchian_upper_aligned[0] = np.nan
    donchian_lower_aligned[0] = np.nan
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if weekly close > EMA50, bearish if weekly close < EMA50
    bullish_regime = weekly_close > ema_50_1w
    bearish_regime = weekly_close < ema_50_1w
    
    # Align regime filters to 1d timeframe
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1w, bullish_regime.astype(float))
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1w, bearish_regime.astype(float))
    
    # Volume confirmation: volume > 2.0 * 20-day average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need Donchian (20-period) and EMA50 (50-period, but aligned)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(bullish_regime_aligned[i]) or np.isnan(bearish_regime_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper AND bullish regime AND volume confirmation
            if close[i] > donchian_upper_aligned[i] and bullish_regime_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND bearish regime AND volume confirmation
            elif close[i] < donchian_lower_aligned[i] and bearish_regime_aligned[i] > 0.5 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian lower (opposite level)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian upper (opposite level)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0