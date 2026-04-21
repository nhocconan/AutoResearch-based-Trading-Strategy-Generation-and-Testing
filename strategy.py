#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_Trend_V1
Hypothesis: Combines ADX (trend strength) with Williams Alligator (trend direction) on 6h timeframe, filtered by 12h HTF trend (EMA50). Enters long when ADX>25 and Alligator bullish (jaw<teeth<lips), short when ADX>25 and Alligator bearish (jaw>teeth>lips). Uses ATR-based trailing stop via signal=0. Designed for low trade frequency in both bull/bear markets by requiring strong trending conditions (ADX>25) to avoid whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for HTF trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams Alligator (13,8,5 SMAs with offsets)
    jaw = pd.Series(close_6h).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, 8 bars ahead
    teeth = pd.Series(close_6h).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, 5 bars ahead
    lips = pd.Series(close_6h).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, 3 bars ahead
    
    # ADX (14-period)
    # True Range
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_6h - np.roll(high_6h, 1))
    down_move = pd.Series(np.roll(low_6h, 1) - low_6h)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) 
            or np.isnan(adx[i]) or np.isnan(atr[i]) 
            or np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        price = close_6h[i]
        
        # Alligator conditions
        alligator_bullish = jaw[i] < teeth[i] and teeth[i] < lips[i]
        alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        
        # HTF trend filter
        htf_uptrend = price > ema_50_12h_aligned[i]
        htf_downtrend = price < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Strong trend + Alligator bullish + HTF uptrend
            if strong_trend and alligator_bullish and htf_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Strong trend + Alligator bearish + HTF downtrend
            elif strong_trend and alligator_bearish and htf_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            
            # ATR trailing stop (2.0 * ATR from high)
            if price < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            # Exit: Alligator turns bearish or trend weakens
            elif not alligator_bullish or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            
            # ATR trailing stop (2.0 * ATR from low)
            if price > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            # Exit: Alligator turns bullish or trend weakens
            elif not alligator_bearish or adx[i] < 20:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_Trend_V1"
timeframe = "6h"
leverage = 1.0