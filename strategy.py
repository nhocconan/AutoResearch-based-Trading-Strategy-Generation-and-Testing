#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_V1
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) filtered by 1d regime (ADX + EMA200). 
Long when Bull Power > 0 and price > EMA200_1d (bull regime). 
Short when Bear Power < 0 and price < EMA200_1d (bear regime). 
Uses volume confirmation (1.5x average) to avoid false signals. 
Designed to work in both bull and bear markets by adapting to regime via EMA200 filter.
Timeframe: 6h, uses 1d HTF for regime filter.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for regime: EMA200 and ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA200 for regime (bull/bear) ===
    df_1d_close = df_1d['close'].values
    ema_200_1d = pd.Series(df_1d_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1d ADX for trend strength (optional filter) ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(df_1d_high - df_1d_low)
    tr2 = pd.Series(np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr3 = pd.Series(np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(np.where(df_1d_high - np.roll(df_1d_high, 1) > np.roll(df_1d_low, 1) - df_1d_low,
                                 np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0))
    dm_minus = pd.Series(np.where(np.roll(df_1d_low, 1) - df_1d_low > df_1d_high - np.roll(df_1d_high, 1),
                                  np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0))
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Elder Ray: Bull Power and Bear Power ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 13-period EMA of close (standard for Elder Ray)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) 
            or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_200 = ema_200_1d_aligned[i]
        adx = adx_1d_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average (moderate filter)
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        # Regime filter: only trade in strong trends (ADX > 20) to avoid whipsaw
        strong_trend = adx > 20
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) AND price > EMA200_1d (bull regime)
            long_condition = (bull > 0) and (price > ema_200) and volume_confirmed and strong_trend
            # Short: Bear Power < 0 (strong selling) AND price < EMA200_1d (bear regime)
            short_condition = (bear < 0) and (price < ema_200) and volume_confirmed and strong_trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime change exit: price crosses below EMA200_1d
            elif price < ema_200:
                signals[i] = 0.0
                position = 0
            # Momentum exit: Bull Power turns negative
            elif bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime change exit: price crosses above EMA200_1d
            elif price > ema_200:
                signals[i] = 0.0
                position = 0
            # Momentum exit: Bear Power turns positive
            elif bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_V1"
timeframe = "6h"
leverage = 1.0