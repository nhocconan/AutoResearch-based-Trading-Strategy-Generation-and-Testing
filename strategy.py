#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w ADX trend strength filter + 1d Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day high with 1w ADX > 25 (strong trend) and volume > 1.5x 20-day volume average.
Short when price breaks below 20-day low with 1w ADX > 25 (strong trend) and volume > 1.5x 20-day volume average.
Exit on opposite Donchian break. Uses discrete position sizing 0.25 to limit fee drag.
ADX filters for strong trending markets only, avoiding whipsaws in ranging conditions. Designed to capture
strong trends in both bull and bear markets while avoiding choppy periods that cause false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend strength
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w ADX(14) for trend strength
    def adx(high_vals, low_vals, close_vals, window):
        plus_dm = np.diff(high_vals, prepend=high_vals[0])
        minus_dm = np.diff(low_vals, prepend=low_vals[0]) * -1
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        tr1 = np.abs(np.subtract(high_vals, low_vals))
        tr2 = np.abs(np.subtract(high_vals, np.append(close_vals[0], close_vals[:-1])))
        tr3 = np.abs(np.subtract(low_vals, np.append(close_vals[0], close_vals[:-1])))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=window, min_periods=window).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=window, min_periods=window).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_vals = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
        return adx_vals
    
    adx_1w = adx(high_1w, low_1w, close_1w, 14)
    
    # Calculate 1d Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_1d, low_1d, 20)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        # Trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above 20-day high with strong trend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                strong_trend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with strong trend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  strong_trend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wADX25_Donchian20_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0