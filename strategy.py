#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and ADX Trend Filter
# Hypothesis: Donchian channel breakouts capture strong momentum moves. 
# Filtered by ADX > 25 to ensure trending markets and volume > 1.5x average 
# to confirm institutional participation. Works in both bull and bear markets 
# by capturing breakouts in the direction of the trend. Target: 20-40 trades/year.
name = "4h_donchian_breakout_volume_adx_v2"
timeframe = "4h"
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
    
    # Get 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    # ADX requires +DI, -DI, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channel (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend weakens
            if close[i] < donchian_low[i] or adx_4h[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend weakens
            if close[i] > donchian_high[i] or adx_4h[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation and strong trend
            if vol_filter[i] and adx_4h[i] > 25:
                # Breakout mode: price breaks Donchian levels
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals