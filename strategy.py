#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d ADX regime filter + 4h Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-period high with ADX > 25 (trending) and volume > 1.5x 20-period volume average.
Short when price breaks below 20-period low with ADX > 25 (trending) and volume > 1.5x 20-period volume average.
Exit when price returns to opposite Donchian boundary or ADX < 20 (range regime).
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
ADX filter avoids whipsaws in ranging markets, Donchian provides structural breakout levels,
volume confirms institutional participation. Works in bull markets (breakout continuation) and 
bear markets (strong trend continuation after ranging periods).
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 1d ADX(14) for regime filter
    def calculate_adx(high_vals, low_vals, close_vals, window=14):
        plus_dm = np.zeros_like(high_vals)
        minus_dm = np.zeros_like(low_vals)
        
        for i in range(1, len(high_vals)):
            up_move = high_vals[i] - high_vals[i-1]
            down_move = low_vals[i-1] - low_vals[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0
        
        tr1 = np.abs(high_vals - low_vals)
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr1[0] = tr2[0] = tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=window, adjust=False, min_periods=window).mean().values / atr
        
        dx = np.abs(plus_di - minus_di) / (np.abs(plus_di) + np.abs(minus_di)) * 100
        dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)
        adx = pd.Series(dx).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_4h, low_4h, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        # Regime filter: ADX > 25 for trending market
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20  # exit condition for ranging
        
        if position == 0:
            # Long: price breaks above 20-period high with trending regime and volume
            if (close[i] > donchian_upper_aligned[i] and 
                trending and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with trending regime and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  trending and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low OR market becomes ranging
            if (close[i] < donchian_lower_aligned[i] or ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-period high OR market becomes ranging
            if (close[i] > donchian_upper_aligned[i] or ranging):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dADX25_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0