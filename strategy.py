#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume and ADX Filter
Hypothesis: Weekly Donchian channels capture long-term market structure. Breakouts
with volume confirmation and ADX > 20 filter capture strong trends in both bull
and bear markets. The 1d timeframe reduces trade frequency to minimize fee drag,
while weekly trend filter ensures we trade with the dominant trend. Volume
confirms institutional participation. This strategy targets 10-25 trades/year.
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    donchian_high = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # ADX filter on weekly (trend strength)
    # Calculate directional movement
    up_move = pd.Series(df_1w['high'].values).diff()
    down_move = -pd.Series(df_1w['low'].values).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # True range
    tr1 = pd.Series(df_1w['high'].values) - pd.Series(df_1w['low'].values)
    tr2 = np.abs(pd.Series(df_1w['high'].values) - pd.Series(df_1w['close'].values).shift(1))
    tr3 = np.abs(pd.Series(df_1w['low'].values) - pd.Series(df_1w['close'].values).shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smoothed values
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smooth / atr_1w
    minus_di = 100 * minus_dm_smooth / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx = adx_1w_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and trend strength
            if price > donchian_high_aligned[i] and vol_ok and adx > 20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and trend strength
            elif price < donchian_low_aligned[i] and vol_ok and adx > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to weekly Donchian low or trend weakens
            if price < donchian_low_aligned[i] or adx < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly Donchian high or trend weakens
            if price > donchian_high_aligned[i] or adx < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0