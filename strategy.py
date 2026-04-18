#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d ADX Trend and Volume
Hypothesis: Breakouts of the 20-period Donchian channel, filtered by 1d ADX > 25 for trend strength and volume > 1.5x 20-period average, capture strong momentum moves in both bull and bear markets. This strategy targets ~25 trades/year to minimize fee drag while capturing significant breakout moves. The ADX filter ensures we only trade in trending conditions, reducing whipsaws during sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14 = adx  # ADX(14)
    
    # Align ADX to 4h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Donchian channel on 4h (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_14_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_14_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: breakout above upper Donchian with volume and strong trend
            if price > highest_high[i] and vol_ok and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian with volume and strong trend
            elif price < lowest_low[i] and vol_ok and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to the middle of the channel or trend weakens
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if price < mid_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to the middle of the channel or trend weakens
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if price > mid_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0