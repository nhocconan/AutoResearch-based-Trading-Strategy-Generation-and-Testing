#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_VolumeFilter_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) filtered by 12h trend regime and volume spike.
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
- Regime: 12h ADX > 25 = trending (follow Elder Ray signals), ADX <= 25 = ranging (fade extremes)
- In trending regime: Long when Bull Power > 0 and rising, Short when Bear Power > 0 and rising
- In ranging regime: Long when Bear Power < 0 and turning up, Short when Bull Power < 0 and turning down
- Volume filter: require 1.5x average volume to avoid low-conviction moves
- Discrete position sizing (0.0, ±0.25) to minimize fee churn
- Designed for 12-37 trades/year (~50-150 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for regime, 1d for EMA13)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 12h ADX for regime detection ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                                 np.maximum(high_12h - np.roll(high_12h, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                                  np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0))
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    tr_smooth = tr_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_12h = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 1d EMA13 for Elder Ray calculation ===
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # === 6h Bull Power and Bear Power ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    bull_power = high - ema_13_1d_aligned  # High - EMA13
    bear_power = ema_13_1d_aligned - low   # EMA13 - Low
    
    # === ATR (21-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=21, min_periods=21).mean().values
    
    # === Volume filter: 1.5x 20-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Power derivatives for momentum
        bull_power_momentum = bull_power[i] - bull_power[i-1] if i > 0 else 0
        bear_power_momentum = bear_power[i] - bear_power[i-1] if i > 0 else 0
        
        # Regime: 12h ADX > 25 = trending, else ranging
        is_trending = adx_12h_aligned[i] > 25
        
        if position == 0:
            if is_trending:
                # Trending regime: follow Elder Ray momentum
                long_signal = bull_power[i] > 0 and bull_power_momentum > 0
                short_signal = bear_power[i] > 0 and bear_power_momentum > 0
            else:
                # Ranging regime: fade extremes (mean reversion)
                long_signal = bear_power[i] < 0 and bear_power_momentum > 0  # Bear power turning up from negative
                short_signal = bull_power[i] < 0 and bull_power_momentum < 0  # Bull power turning down from negative
            
            # Volume confirmation
            if long_signal and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss: 2.5x ATR
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions
            elif is_trending:
                # In trending regime, exit when Elder Ray momentum fades
                long_exit = bull_power[i] <= 0 or bull_power_momentum <= 0
            else:
                # In ranging regime, exit when mean reversion completes
                long_exit = bear_power[i] >= 0  # Bear power back to zero or positive
            
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss: 2.5x ATR
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions
            elif is_trending:
                # In trending regime, exit when Elder Ray momentum fades
                short_exit = bear_power[i] <= 0 or bear_power_momentum <= 0
            else:
                # In ranging regime, exit when mean reversion completes
                short_exit = bull_power[i] >= 0  # Bull power back to zero or positive
            
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0