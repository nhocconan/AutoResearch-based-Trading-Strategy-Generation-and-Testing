#!/usr/bin/env python3
# 6h_LongTermChannel_FilteredBreakout_v1
# Hypothesis: Uses 20-period 6h Donchian channel breakouts with 1w ADX trend filter and volume confirmation.
# Designed for low trade frequency (target 15-30/year) to minimize fee drag. Works in bull/bear by aligning with 1w trend.
# Position size 0.25 for balanced risk. Focus on BTC/ETH robustness.

name = "6h_LongTermChannel_FilteredBreakout_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX on 1w (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    tr_smooth = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1w ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Warmup for Donchian and 1w data
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade in strong trends (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: break above Donchian high with volume and trend
            if close[i] > donchian_high[i] and volume_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: break below Donchian low with volume and trend
            elif close[i] < donchian_low[i] and volume_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Donchian low or trend weakens
            if close[i] < donchian_low[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian high or trend weakens
            if close[i] > donchian_high[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals