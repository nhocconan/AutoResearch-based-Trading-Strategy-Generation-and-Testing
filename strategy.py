#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Camarilla R1/S1 breakout + volume confirmation + ADX regime filter.
Long when price breaks above 12h Camarilla R1 with volume > 1.2x 20-period average and ADX < 25 (range market).
Short when price breaks below 12h Camarilla S1 with volume > 1.2x 20-period average and ADX < 25.
Exit when price returns to the 12h Camarilla midpoint (R1+S1)/2.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Works in bull markets (breakouts in ranging conditions) and bear markets (mean reversion after failed breakouts).
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
    
    # Get 12h data for Camarilla levels, volume, and ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla levels (based on previous day's range)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use the previous 12h bar's high/low/close to calculate levels for current bar
    prev_high_12h = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low_12h = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    
    R1 = prev_close_12h + 1.1 * (prev_high_12h - prev_low_12h) / 12
    S1 = prev_close_12h - 1.1 * (prev_high_12h - prev_low_12h) / 12
    midpoint = (R1 + S1) / 2
    
    # Calculate 12h ADX (14-period) for regime filter
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 12h volume 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    midpoint_aligned = align_htf_to_ltf(prices, df_12h, midpoint)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.2x 20-period average
        volume_confirmed = volume_12h_aligned[i] > 1.2 * vol_ma_20_12h_aligned[i]
        
        # Regime filter: ADX < 25 (range market - good for mean reversion at pivot levels)
        regime_filter = adx_aligned[i] < 25
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R1 with volume and in range market
            if (close[i] > R1_aligned[i] and 
                volume_confirmed and 
                regime_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S1 with volume and in range market
            elif (close[i] < S1_aligned[i] and 
                  volume_confirmed and 
                  regime_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 12h Camarilla midpoint
            if close[i] < midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 12h Camarilla midpoint
            if close[i] > midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hCamarilla_R1S1_Volume_ADXRange"
timeframe = "4h"
leverage = 1.0