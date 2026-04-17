#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike.
Long when price breaks above 1d Donchian upper AND 1w close > EMA34 AND 1d volume > 1.5x 20-bar average.
Short when price breaks below 1d Donchian lower AND 1w close < EMA34 AND 1d volume > 1.5x 20-bar average.
Exit when price touches opposite Donchian band or 1w EMA34 crosses in opposite direction.
Uses 1d for breakout and volume, 1w for trend filter. Target: 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian(20)
    donch_h = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 1d timeframe (primary timeframe)
    donch_h_aligned = align_htf_to_ltf(prices, df_1d, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_1d, donch_l)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_h_aligned[i]) or 
            np.isnan(donch_l_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-bar average
        volume_confirmed = volume_1d[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: 1w close relative to EMA34
        uptrend = close_1d[i] > ema_34_1w_aligned[i]
        downtrend = close_1d[i] < ema_34_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = close_1d[i] > donch_h_aligned[i]
        breakout_dn = close_1d[i] < donch_l_aligned[i]
        
        # Exit conditions: touch opposite band or trend reversal
        exit_long = close_1d[i] < donch_l_aligned[i] or not uptrend
        exit_short = close_1d[i] > donch_h_aligned[i] or not downtrend
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and uptrend
            if (breakout_up and volume_confirmed and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and downtrend
            elif (breakout_dn and volume_confirmed and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch Donchian low or trend turns down
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch Donchian high or trend turns up
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_1wEMA34_Trend"
timeframe = "1d"
leverage = 1.0