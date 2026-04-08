#!/usr/bin/env python3
# 4h_donchian_volume_adx_v2
# Hypothesis: Donchian channel breakouts on 4h timeframe with volume confirmation and ADX trend filter capture strong momentum moves. Works in both bull and bear markets by requiring volume > 1.5x 20-period average and ADX > 25 to filter weak breakouts. Position size 0.25 to manage risk during drawdowns. Target: 20-50 trades/year per symbol.

name = "4h_donchian_volume_adx_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate ADX on daily timeframe (trend strength filter)
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_d - np.roll(high_d, 1)
    down_move = np.roll(low_d, 1) - low_d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Calculate Donchian channels on 4h data
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 20-period average volume for 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = donchian_period
    
    for i in range(start_idx, n):
        # Get aligned daily ADX for current 4h bar
        adx_val = align_htf_to_ltf(prices, df_d, adx)[i]
        
        # Skip if any required data is NaN
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_val) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 1.5x 20-period average
        vol_breakout = volume[i] > 1.5 * vol_ma[i]
        
        # Strong trend condition: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 1:  # Long position
            # Exit if price breaks below lower Donchian channel
            if close[i] < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above upper Donchian channel
            if close[i] > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above upper channel with volume confirmation and strong trend
            if high[i] >= upper_channel[i] and close[i] > upper_channel[i] and vol_breakout and strong_trend:
                position = 1
                signals[i] = 0.25
            # Breakout short below lower channel with volume confirmation and strong trend
            elif low[i] <= lower_channel[i] and close[i] < lower_channel[i] and vol_breakout and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals