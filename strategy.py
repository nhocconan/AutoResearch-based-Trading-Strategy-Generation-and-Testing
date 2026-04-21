#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_RegimeFilter_v1
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) combined with 1d ADX regime filter.
In strong trends (ADX_1d > 25): trade in direction of Elder Ray (long if Bull Power > 0, short if Bear Power < 0).
In ranging markets (ADX_1d < 20): fade extreme Elder Ray readings (long if Bull Power < -std, short if Bear Power > +std).
Uses discrete sizing (0.25) and ATR(14) stoploss (2.5x). Designed for 6h timeframe to target 50-150 trades over 4 years (12-37/year).
Works in bull/bear via regime adaptation - trend following in trends, mean reversion in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for ADX regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d OHLC for ADX regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    
    # Smoothed values
    tr_14 = tr_1d.rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = dm_plus.rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = dm_minus.rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h EMA13 for Elder Ray calculation ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volatility normalization for mean reversion thresholds
    atr_6h = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    bull_power_std = pd.Series(bull_power).rolling(window=50, min_periods=20).std().values
    bear_power_std = pd.Series(bear_power).rolling(window=50, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(atr_6h[i]) 
            or np.isnan(bull_power_std[i]) or np.isnan(bear_power_std[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_regime = adx_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        bp_std = bull_power_std[i]
        br_std = bear_power_std[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx_regime > 25:  # Strong trend - trend following
                long_condition = bp > 0  # Bull power positive
                short_condition = br < 0  # Bear power negative
            elif adx_regime < 20:  # Ranging market - mean reversion
                long_condition = bp < -0.5 * bp_std  # Extremely weak bull power
                short_condition = br > 0.5 * br_std  # Extremely strong bear power
            else:  # Transition regime - no entries
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (for trend regime) or mean reversion exit (for range regime)
            elif adx_regime > 25 and bp < 0:  # Trend regime: exit when bull power turns negative
                signals[i] = 0.0
                position = 0
            elif adx_regime < 20 and bp > 0.5 * bp_std:  # Range regime: exit when bull power recovers
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (for trend regime) or mean reversion exit (for range regime)
            elif adx_regime > 25 and br > 0:  # Trend regime: exit when bear power turns positive
                signals[i] = 0.0
                position = 0
            elif adx_regime < 20 and br < -0.5 * br_std:  # Range regime: exit when bear power recovers
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0