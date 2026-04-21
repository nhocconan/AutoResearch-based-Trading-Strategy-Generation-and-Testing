#!/usr/bin/env python3
"""
12h_1d_Donchian20_Breakout_Volume_Regime_Filtered_v1
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 12h ADX regime filter.
Breakouts above 12h upper band go long; below lower band go short.
Volume filter requires 1d volume > 1.5x 20-day average.
ADX filter: only trade when ADX(14) > 25 (trending market).
Exit when price crosses 12h EMA(34) or after 3 bars.
Designed to work in both bull and bear by capturing strong trending moves with volume confirmation.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Donchian and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) on 12h
    if len(high_12h) >= 20:
        upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    else:
        upper_12h = np.full_like(high_12h, np.nan)
        lower_12h = np.full_like(low_12h, np.nan)
    
    # Calculate EMA(34) on 12h for exit
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ADX(14) on 12h for regime filter
    if len(high_12h) >= 14:
        # True Range
        tr1 = high_12h[1:] - low_12h[1:]
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
        dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    else:
        adx = np.full_like(high_12h, np.nan)
    
    # Align 12h indicators to 12h timeframe (no alignment needed as we're already on 12h)
    upper_12h_aligned = upper_12h
    lower_12h_aligned = lower_12h
    ema_34_12h_aligned = ema_34_12h
    adx_aligned = adx
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / (vol_ma_1d + 1e-10)
    
    # Align 1d volume ratio to 12h timeframe
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        regime_ok = adx_aligned[i] > 25
        
        # Volume filter: 1d volume > 1.5x 20-day average
        volume_ok = vol_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long entry: price breaks above upper Donchian band with volume and regime
            if price > upper_12h_aligned[i] and volume_ok and regime_ok:
                signals[i] = 0.25
                position = 1
                bars_in_trade = 0
            # Short entry: price breaks below lower Donchian band with volume and regime
            elif price < lower_12h_aligned[i] and volume_ok and regime_ok:
                signals[i] = -0.25
                position = -1
                bars_in_trade = 0
        
        elif position != 0:
            bars_in_trade += 1
            
            # Exit conditions:
            # 1. Price crosses EMA(34)
            # 2. Maximum 3 bars in trade (to avoid overtrading)
            ema_cross = (position == 1 and price < ema_34_12h_aligned[i]) or \
                        (position == -1 and price > ema_34_12h_aligned[i])
            max_bars = bars_in_trade >= 3
            
            if ema_cross or max_bars:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_1d_Donchian20_Breakout_Volume_Regime_Filtered_v1"
timeframe = "12h"
leverage = 1.0