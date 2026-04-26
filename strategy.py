#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dADX_Trend_v1
Hypothesis: Camarilla R4/S4 breakout on 6h with 1d ADX(14) > 25 trend filter and volume > 1.2x median. Targets R4/S4 as strong breakout levels where price often continues with momentum. Uses 1d ADX to ensure trading only in trending markets (both bull/bear) and volume filter for conviction. Designed to avoid ranging markets and false breakouts. Targets 12-37 trades/year via tight entry conditions (R4/S4 breakouts are rarer than R3/S3).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend (ADX)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX(14) for trend strength filter
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Calculate Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smooth TR and DM
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 6h data for Camarilla levels (using previous 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    cam_high = pd.Series(df_6h['high'].values).shift(1).values
    cam_low = pd.Series(df_6h['low'].values).shift(1).values
    cam_close = pd.Series(df_6h['close'].values).shift(1).values
    
    # Camarilla R4, S4 levels (strong breakout levels)
    R4 = cam_close + (cam_high - cam_low) * 1.1 / 2
    S4 = cam_close - (cam_high - cam_low) * 1.1 / 2
    
    # Volume filter: volume > 1.2x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    R4_aligned = align_htf_to_ltf(prices, df_6h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_6h, S4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of ADX(14) 1d, Camarilla (need 2 bars for shift), volume median (20), ATR (14)
    start_idx = max(14, 2, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx_val = adx_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r4_val = R4_aligned[i]
        s4_val = S4_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market (strong enough for breakout)
        trending = adx_val > 25
        
        # Volume filter: only trade in above-average volume environments
        volume_filter = volume_val > 1.2 * vol_median_val
        
        if position == 0:
            # Long: break above R4 with volume filter, and trending market
            long_signal = (close_val > r4_val) and \
                          volume_filter and \
                          trending
            
            # Short: break below S4 with volume filter, and trending market
            short_signal = (close_val < s4_val) and \
                           volume_filter and \
                           trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dADX_Trend_v1"
timeframe = "6h"
leverage = 1.0