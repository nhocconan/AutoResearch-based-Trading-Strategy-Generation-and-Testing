#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-day ADX(14) trend filter and volume confirmation.
# The 1-day ADX > 25 identifies trending markets, avoiding whipsaws in ranges.
# Donchian(20) breakout captures momentum in the direction of the daily trend.
# Volume > 1.5x the 20-period average confirms institutional participation.
# Exit occurs when ADX drops below 20 (trend weakening) or price returns to the midpoint of the Donchian channel.
# This combination aims for 15-30 trades per year per symbol (60-120 total over 4 years), staying within the optimal range to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on daily data
    adx_len = 14
    if len(df_1d) < adx_len * 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    
    # Align ADX to 6h timeframe (wait for completed daily candle)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channel (20 periods) on 6h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    dc_mid = (dc_upper + dc_lower) / 2
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(adx_len * 2, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        weakening_trend = adx_aligned[i] < 20  # exit when trend weakens
        
        if position == 0:
            # Enter long: Donchian breakout above + strong trend + volume
            if (close[i] > dc_upper[i] and 
                strong_trend and 
                volume[i] > 1.5 * vol_ma[i]):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + strong trend + volume
            elif (close[i] < dc_lower[i] and 
                  strong_trend and 
                  volume[i] > 1.5 * vol_ma[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakening OR price returns to Donchian midpoint
            if weakening_trend or close[i] < dc_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakening OR price returns to Donchian midpoint
            if weakening_trend or close[i] > dc_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ADX_Donchian_Volume_v1"
timeframe = "6h"
leverage = 1.0