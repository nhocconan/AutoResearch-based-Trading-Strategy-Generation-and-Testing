#!/usr/bin/env python3
# 4h_donchian_1d_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + CHOP > 61.8 (range).
# Short when price breaks below Donchian(20) low + volume confirmation + CHOP > 61.8.
# Uses discrete sizing (±0.25) to minimize fee drag. Designed for low trade frequency (target: 75-200 trades over 4 years).
# Works in bull/bear markets by using choppiness filter to avoid whipsaws in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR (14-period) for choppiness
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ADX (14-period) for choppiness
    up_move = pd.Series(high_1d) - pd.Series(high_1d).shift(1)
    down_move = pd.Series(low_1d).shift(1) - pd.Series(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * plus_dm_smooth / (atr_1d + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1d + 1e-10)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # Simplified: CHOP = 100 * log10(atr_1d_sum / (rolling_max - rolling_min)) / log10(14)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1d volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume_1d > 1.5 * volume_ma
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed.astype(float))
    
    # 4h Donchian channels (20-period)
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_confirmed_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine breakout conditions
        breakout_up = close[i] > dc_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < dc_low[i-1]  # Break below previous Donchian low
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel OR breakout down
            if close[i] < dc_high[i] or breakout_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel OR breakout up
            if close[i] > dc_low[i] or breakout_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if ranging and volume_confirmed_aligned[i] > 0.5:  # Volume confirmed
                if breakout_up:
                    position = 1
                    signals[i] = 0.25
                elif breakout_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals