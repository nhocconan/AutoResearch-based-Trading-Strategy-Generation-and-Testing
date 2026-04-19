#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with weekly pivot point resistance/support breakouts,
# filtered by weekly ADX trend strength and volume confirmation.
# Weekly pivots provide robust support/resistance levels that work in both bull and bear markets.
# ADX filter ensures we only trade in trending conditions, reducing whipsaws.
# Volume confirmation adds conviction to breakouts.
# Target: 20-40 trades/year to avoid fee drag.

name = "12h_1w_PivotPoint_R1S1_Breakout_ADX_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly ADX(14) for trend strength
    # TR
    tr1 = np.maximum(high_1w[1:], close_1w[:-1]) - np.minimum(low_1w[1:], close_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # +DM and -DM
    up_move = np.concatenate([[0], high_1w[1:] - high_1w[:-1]])
    down_move = np.concatenate([[0], low_1w[:-1] - low_1w[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    # DI and DX
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Weekly Pivot Points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    prev_high = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close = np.concatenate([[np.nan], close_1w[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align weekly data to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Allow warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_aligned[i] > 20
        
        if position == 0:
            # Long: price breaks above R1 with volume and trending market
            if price > r1_aligned[i] and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trending market
            elif price < s1_aligned[i] and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below pivot or ADX weakens
            if price < pivot[i] or adx_aligned[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above pivot or ADX weakens
            if price > pivot[i] or adx_aligned[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals