# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with volume confirmation and weekly ADX trend filter.
In low volatility (range) markets, false breakouts are filtered by weak volume and weak trend.
In high volatility (trend) markets, strong volume and strong ADX confirm genuine breakouts.
Works in both bull and bear markets by capturing momentum after consolidation.
Target: 15-25 trades/year per symbol (<100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly ADX(14) for trend strength ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === 12h Donchian Channel (20-period) ===
    high_12h = prices['high'].rolling(window=20, min_periods=20).max().values
    low_12h = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(high_12h[i]) or 
            np.isnan(low_12h[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        adx_val = adx_aligned[i]
        upper_channel = high_12h[i]
        lower_channel = low_12h[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long on upper breakout with strong trend and volume
            if (price_close > upper_channel and
                adx_val > 25 and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short on lower breakdown with strong trend and volume
            elif (price_close < lower_channel and
                  adx_val > 25 and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when trend weakens or price returns to channel
            if position == 1 and (adx_val < 20 or price_close < upper_channel):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (adx_val < 20 or price_close > lower_channel):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_ADXTrend_Volume"
timeframe = "12h"
leverage = 1.0