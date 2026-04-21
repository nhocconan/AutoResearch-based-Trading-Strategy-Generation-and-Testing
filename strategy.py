#!/usr/bin/env python3
"""
6h_ElderRay_Regime_ADX_v1
Hypothesis: On 6h timeframe, Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with ADX regime filter (ADX > 25 for trending, ADX < 20 for range) captures strong directional moves with reduced whipsaw. In trending regime (ADX > 25), trade in direction of Elder Ray (long if Bull Power > 0, short if Bear Power > 0). In range regime (ADX < 20), fade extreme Elder Ray values (long if Bear Power < 0 and Bull Power < threshold, short if Bull Power > 0 and Bear Power < threshold). Uses 1d EMA50 for higher timeframe trend filter to avoid counter-trend trades. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for higher timeframe trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Elder Ray Index (EMA13) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    
    # === 6h ADX (14) for regime filter ===
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 8  # max 2 days (8 * 6h = 48h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        htf_ema = ema_50_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        adx_val = adx[i]
        ema13_val = ema_13[i]
        
        # Regime filters
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        is_bull_htf = price > htf_ema
        is_bear_htf = price < htf_ema
        
        if position == 0:
            if is_trending:
                # Trending regime: trade with Elder Ray direction
                long_condition = (bull > 0) and is_bull_htf
                short_condition = (bear > 0) and is_bear_htf
            elif is_ranging:
                # Ranging regime: fade extreme Elder Ray
                # Long when bear power is negative and bull power is not too extreme
                long_condition = (bear < 0) and (bull < 0.5 * np.std(bull_power[max(0, i-50):i+1])) and is_bull_htf
                # Short when bull power is positive and bear power is not too extreme
                short_condition = (bull > 0) and (bear < 0.5 * np.std(bear_power[max(0, i-50):i+1])) and is_bear_htf
            else:
                # Transition regime (ADX between 20-25): require stronger signals
                long_condition = (bull > 0) and (bull > 0.3 * np.std(bull_power[max(0, i-50):i+1])) and is_bull_htf
                short_condition = (bear > 0) and (bear > 0.3 * np.std(bear_power[max(0, i-50):i+1])) and is_bear_htf
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_ADX_v1"
timeframe = "6h"
leverage = 1.0