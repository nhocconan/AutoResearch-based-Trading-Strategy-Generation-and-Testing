#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_RangeFilter
Hypothesis: Breakouts from weekly Donchian channels with 1d range filter (ADX<25) capture strong trends while avoiding chop. Works in bull via upside breakouts, bear via downside breakdowns. Volume confirmation filters weak moves. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high, prepend=high[0]))
    tr2 = np.abs(np.diff(low, prepend=low[0]))
    tr3 = np.abs(np.diff(close, prepend=close[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean() / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean()
    return adx.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max()
    lower = pd.Series(low).rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-period)
    upper_1w, lower_1w = calculate_donchian(high_1w, low_1w, 20)
    
    # Align weekly Donchian to daily
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Daily ADX for range filter (ADX < 25 = ranging, avoid breakouts in chop)
    adx_1d = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: break above weekly upper Donchian in low volatility (trending conditions)
        long_condition = (close[i] > upper_1w_aligned[i]) and (adx_1d[i] < 25) and volume_expansion[i]
        
        # Short: break below weekly lower Donchian in low volatility (trending conditions)
        short_condition = (close[i] < lower_1w_aligned[i]) and (adx_1d[i] < 25) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        else:
            # Exit: reverse signal or volatility expansion (chop detection)
            if position == 1 and (adx_1d[i] >= 25 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (adx_1d[i] >= 25 or not volume_expansion[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Donchian_Breakout_RangeFilter"
timeframe = "1d"
leverage = 1.0