#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_ADX
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with ADX regime filter on 6h timeframe. 
Long when Bull Power > 0 and ADX > 25 (strong trend) and EMA13 rising. 
Short when Bear Power > 0 and ADX > 25 and EMA13 falling. 
Uses 1d HTF trend filter (EMA50) to avoid counter-trend trades. 
Designed for low trade frequency (12-37/year) via strict trend + power conditions. 
Discrete sizing 0.25 minimizes fee churn. Works in bull/bear via ADX regime and HTF trend alignment.
"""

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
    
    # Calculate EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Calculate ADX(14)
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = np.nan
    down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 1d HTF data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 14 for ADX/TR, 13 for EMA13, 50 for 1d EMA50
    start_idx = max(14, 13, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or
            np.isnan(plus_di[i]) or
            np.isnan(minus_di[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema13_val = ema13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        adx_val = adx[i]
        plus_di_val = plus_di[i]
        minus_di_val = minus_di[i]
        ema_50_val = ema_50_1d_aligned[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: Bull Power > 0, ADX > 25, +DI > -DI (bullish momentum), price > EMA50_1d (uptrend filter)
            long_entry = (bull_val > 0) and (adx_val > 25) and (plus_di_val > minus_di_val) and (close_val > ema_50_val)
            # Short: Bear Power > 0, ADX > 25, -DI > +DI (bearish momentum), price < EMA50_1d (downtrend filter)
            short_entry = (bear_val > 0) and (adx_val > 25) and (minus_di_val > plus_di_val) and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on weakening bull power, ADX drop, or EMA13 cross below
            if (bull_val <= 0) or (adx_val < 20) or (close_val < ema13_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on weakening bear power, ADX drop, or EMA13 cross above
            if (bear_val <= 0) or (adx_val < 20) or (close_val > ema13_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_ADX"
timeframe = "6h"
leverage = 1.0