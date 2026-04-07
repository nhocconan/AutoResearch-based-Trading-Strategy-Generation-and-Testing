#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Daily Volume and ADX Filter
# Hypothesis: Breakouts from 20-period Donchian channels on 12h timeframe,
# filtered by daily volume surge and ADX trend strength, work in both bull and bear markets
# by capturing momentum bursts. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_donchian20_volume_adx_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for filters
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Calculate daily ADX (14-period) for trend strength
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Directional Movement
    up_move = high_daily - np.roll(high_daily, 1)
    down_move = np.roll(low_daily, 1) - low_daily
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx[0:13] = np.nan  # First 13 values invalid
    
    adx_12h = align_htf_to_ltf(prices, df_daily, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma_12h[i]) or np.isnan(adx_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x daily average
        vol_surge = volume[i] > (1.5 * vol_ma_12h[i])
        
        # ADX filter: trend strength > 25
        strong_trend = adx_12h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price touches opposite band or trend weakens
            if low[i] <= low_min[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches opposite band or trend weakens
            if high[i] >= high_max[i] or adx_12h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout with volume surge and strong trend
            if vol_surge and strong_trend:
                if high[i] >= high_max[i]:  # Break above upper band
                    position = 1
                    signals[i] = 0.25
                elif low[i] <= low_min[i]:  # Break below lower band
                    position = -1
                    signals[i] = -0.25
    
    return signals