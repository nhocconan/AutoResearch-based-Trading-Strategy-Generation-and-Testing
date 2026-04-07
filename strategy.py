#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) + Volume + ADX Filter
# Hypothesis: Breakouts of 20-period Donchian channels on 12h timeframe, filtered by
# volume confirmation and ADX trend strength, capture sustained directional moves.
# Works in bull via upper band breakouts, in bear via lower band breakdowns, and avoids
# whipsaws in ranging markets via ADX filter. Target: 20-40 trades/year.
name = "12h_donchian20_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 12h alignment
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on daily data (20-period)
    # Upper band = highest high of last 20 days
    # Lower band = lowest low of last 20 days
    high_20d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, high_20d)
    lower_12h = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # ADX filter on daily data (14-period)
    # Calculate +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
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
    
    # Align ADX to 12h
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 12h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or
            np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        # ADX filter: trend strength > 25
        adx_filter = adx_12h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian band
            if close[i] < lower_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian band
            if close[i] > upper_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above upper Donchian band with volume and ADX filter
            if close[i] > upper_12h[i] and vol_confirm and adx_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below lower Donchian band with volume and ADX filter
            elif close[i] < lower_12h[i] and vol_confirm and adx_filter:
                position = -1
                signals[i] = -0.25
    
    return signals