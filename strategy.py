#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Uses Donchian(20) breakouts for entry, 1d ADX(14) > 25 to filter for trending markets,
# and volume > 1.5x 20-period average for confirmation. Exits when price crosses the
# opposite Donchian band. Designed for low trade frequency (12-37/year) to minimize fee drag.
# Works in trending markets via breakouts and avoids choppy markets via ADX filter.

name = "12h_donchian20_1d_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) on 1d data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) > 0, dx, 0)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        if adx_1d_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x 20-period average
        if volume[i] <= 1.5 * vol_ma[i]:
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian high
        if close[i] > donchian_high[i]:
            signals[i] = 0.25
        # Short entry: price breaks below Donchian low
        elif close[i] < donchian_low[i]:
            signals[i] = -0.25
        # Exit: price crosses the opposite Donchian band
        elif (signals[i-1] > 0 and close[i] < donchian_low[i]) or \
             (signals[i-1] < 0 and close[i] > donchian_high[i]):
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = signals[i-1]
    
    return signals