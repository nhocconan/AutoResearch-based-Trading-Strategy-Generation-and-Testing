#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) filtered by 1d EMA50 trend and volume confirmation.
Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price > EMA50_1d.
Short when Bear Power < 0 and falling, Bull Power > 0 and rising, price < EMA50_1d.
Volume confirmation (1.5x average) filters weak entries. Works in both bull and bear markets by requiring trend alignment.
Timeframe: 6h, uses 1d HTF for trend filter and EMA13 for Elder Ray.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend and EMA13 for Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for EMA50 trend ===
    df_1d_close = df_1d['close'].values
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d OHLC for EMA13 (used in Elder Ray) ===
    ema_13_1d = pd.Series(df_1d_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # === 1d OHLC for Elder Ray calculation ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = df_1d_high - ema_13_1d
    bear_power = ema_13_1d - df_1d_low
    
    # Align 1d Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) 
            or np.isnan(bear_power_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average (moderate filter)
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        # Momentum confirmation: Bull Power rising, Bear Power falling
        if i >= 61:
            bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
            bear_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
        else:
            bull_rising = False
            bear_falling = False
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power falling, price > EMA50_1d, volume confirmed
            long_condition = (bull > 0) and bull_rising and bear_falling and (price > ema_trend) and volume_confirmed
            # Short: Bear Power < 0 and falling, Bull Power rising, price < EMA50_1d, volume confirmed
            short_condition = (bear > 0) and bear_falling and bull_rising and (price < ema_trend) and volume_confirmed
            
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
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Elder Ray weakening exit: Bull Power <= 0 or not rising
            elif bull <= 0 or not bull_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Elder Ray weakening exit: Bear Power <= 0 or not falling
            elif bear <= 0 or not bear_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0