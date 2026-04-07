#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with 12-hour ADX trend filter and 1-day volume confirmation
# Uses Donchian(20) breakouts for entries, 12h ADX > 25 for trend strength, and 1d volume > 20-period average
# Designed for moderate trade frequency (target: 15-30 trades/year) with strong trend following
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns

name = "6h_donchian20_12h_adx_1d_volume_v1"
timeframe = "6h"
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
    
    # 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX(14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 1d volume average
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF data to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 1d average volume (scaled)
        vol_condition = volume[i] > (vol_avg_1d_aligned[i] * 1.5)
        
        # Trend condition: ADX > 25
        trend_condition = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > highest_20[i-1]  # Break above previous high
        breakout_down = close[i] < lowest_20[i-1]  # Break below previous low
        
        # Long: upward breakout with trend and volume
        if breakout_up and trend_condition and vol_condition:
            signals[i] = 0.25
        # Short: downward breakout with trend and volume
        elif breakout_down and trend_condition and vol_condition:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals