#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_RegimeFilter_V1
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with regime filter (ADX > 25 for trending, < 20 for ranging). 
In trending regimes (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising. 
In ranging regimes (ADX < 20): fade extremes - long when Bear Power < -0.5*ATR and turning up, short when Bull Power < -0.5*ATR and turning down.
Uses 1w HTF for major trend filter (only take longs above 1w EMA50, shorts below).
Volume confirmation (1.5x average) required for all entries to avoid false signals.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Works in both bull and bear markets via regime adaptation and 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for major trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Load 1d data for ADX regime calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for major trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d ADX for regime filter (14-period) ===
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
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Elder Ray calculations ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    
    # Rate of change for power signals (3-period)
    bull_power_roc = pd.Series(bull_power).diff(3).values
    bear_power_roc = pd.Series(bear_power).diff(3).values
    
    # === 6h ATR for volatility normalization and stops ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_roc[i]) or np.isnan(bear_power_roc[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_trend_1w = ema_50_1w_aligned[i]
        adx_val = adx_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        bp_roc = bull_power_roc[i]
        br_roc = bear_power_roc[i]
        vol_avg = vol_ma[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        # Regime classification
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Entry logic based on regime
            if is_trending:
                # Trending regime: follow Elder Ray momentum with 1w trend filter
                long_condition = (bp > 0) and (bp_roc > 0) and (price > ema_trend_1w) and volume_confirmed
                short_condition = (br > 0) and (br_roc > 0) and (price < ema_trend_1w) and volume_confirmed
            elif is_ranging:
                # Ranging regime: fade Elder Ray extremes
                long_condition = (bp < -0.5 * atr_val) and (bp_roc > 0) and volume_confirmed
                short_condition = (br < -0.5 * atr_val) and (br_roc > 0) and volume_confirmed
            else:
                # Transition regime (ADX 20-25): require stronger signals
                long_condition = (bp > 0) and (bp_roc > 0) and (price > ema_trend_1w) and volume_confirmed
                short_condition = (br > 0) and (br_roc > 0) and (price < ema_trend_1w) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            # Stoploss: 2.5x ATR
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Trend reversal: Elder Ray turns negative
            elif bp < 0:
                signals[i] = 0.0
                position = 0
            # 1w trend filter failure
            elif price < ema_trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            # Stoploss: 2.5x ATR
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Trend reversal: Elder Ray turns negative
            elif br < 0:
                signals[i] = 0.0
                position = 0
            # 1w trend filter failure
            elif price > ema_trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_RegimeFilter_V1"
timeframe = "6h"
leverage = 1.0