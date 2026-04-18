#!/usr/bin/env python3
"""
4h EMA Trend with Volume Confirmation and 12h ADX Filter
Hypothesis: In trending markets (12h ADX > 25), price staying above/below 4h EMA34 with 
volume > 1.5x 20-period EMA indicates strong momentum. Mean reversion when price 
crosses EMA34. Trend filter prevents whipsaws in chop. Works in bull/bear via 
higher timeframe trend confirmation. Target: 25-40 trades/year.
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
    
    # 4h EMA34 for trend
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    # Get 12h ADX for trend filter (trending when > 25)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for ADX
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (using EMA as approximation)
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h_raw = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34[i]) or np.isnan(vol_ratio[i]) or np.isnan(adx_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema34[i]
        vol_conf = vol_ratio[i] > 1.5
        adx_val = adx_12h[i]
        
        if position == 0:
            # Only trade when 12h trend is strong (ADX > 25)
            if adx_val > 25 and vol_conf:
                if price > ema:
                    signals[i] = 0.25
                    position = 1
                elif price < ema:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if trend weakens or price crosses back below EMA
            if adx_val < 20 or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if trend weakens or price crosses back above EMA
            if adx_val < 20 or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA_Trend_Volume_12hADX"
timeframe = "4h"
leverage = 1.0