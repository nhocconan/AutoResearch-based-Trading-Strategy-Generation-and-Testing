#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with 1d volume confirmation and 1d ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (especially L3/H3) act as strong support/resistance. 
# Breakouts with volume confirmation and aligned higher timeframe trend (ADX > 25) 
# capture sustained moves. Works in bull/bear by trading with the trend on breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d indicators for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX (14-period)
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False).mean()
    plus_dm14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean()
    minus_dm14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean()
    adx_1d = adx.values
    
    # 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate Camarilla levels for each 4h bar using previous day's OHLC
    # We'll calculate these inside the loop using rolling window of 1d data
    # but we need to access previous day's close, high, low
    
    # For efficiency, we'll compute daily OHLC and use it to calculate levels
    # Create arrays to store daily OHLC for each 4h bar
    prev_close_1d = np.full(n, np.nan)
    prev_high_1d = np.full(n, np.nan)
    prev_low_1d = np.full(n, np.nan)
    
    # We'll fill these by looking back to the previous day's data
    # Since we have 1d data aligned, we can use the aligned close/high/low
    # but shifted by 1 day
    
    # Get aligned 1d OHLC
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Previous day's values (shift by 1 day = 6 bars in 4h timeframe)
    prev_close_1d = np.roll(close_1d_aligned, 6)
    prev_high_1d = np.roll(high_1d_aligned, 6)
    prev_low_1d = np.roll(low_1d_aligned, 6)
    # First 6 bars will be NaN (no previous day)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(prev_close_1d[i]) or np.isnan(prev_high_1d[i]) or np.isnan(prev_low_1d[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels from previous day's OHLC
        # Camarilla formulas:
        # H4 = close + 1.5 * (high - low)
        # H3 = close + 1.0 * (high - low)
        # L3 = close - 1.0 * (high - low)
        # L4 = close - 1.5 * (high - low)
        daily_range = prev_high_1d[i] - prev_low_1d[i]
        if daily_range <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        h3 = prev_close_1d[i] + 1.0 * daily_range
        l3 = prev_close_1d[i] - 1.0 * daily_range
        
        # Volume confirmation: current volume > 1.5x average daily volume
        vol_confirm = volume[i] > 1.5 * avg_vol_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Entry conditions
        # Long: Price breaks above H3 with volume and trend
        if close[i] > h3 and vol_confirm and trending and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price breaks below L3 with volume and trend
        elif close[i] < l3 and vol_confirm and trending and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to previous day's close (mean reversion) or trend weakens
        elif position == 1 and (close[i] < prev_close_1d[i] or adx_1d_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > prev_close_1d[i] or adx_1d_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals