#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime_VolumeSpike_v1
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 regime filter and volume confirmation (>1.8x 20-period MA).
Elder Ray measures bull/bear strength relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
Long when Bull Power > 0 and rising (2-bar momentum) + price > 1d EMA50 + volume spike.
Short when Bear Power < 0 and falling (2-bar momentum) + price < 1d EMA50 + volume spike.
Uses ATR-based stop (2.0x) and minimum holding period of 2 bars to reduce churn.
Designed for 6h timeframe with 1d HTF regime to work in both bull and bear markets by requiring alignment with higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h EMA13 for Elder Ray calculation ===
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.8x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Elder Ray: Bull Power and Bear Power ===
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume confirmation: current volume > 1.8x average (stricter threshold)
        volume_confirm = volume_now > 1.8 * vol_avg
        
        # Elder Ray momentum: 2-bar change
        if i >= 2:
            bull_momentum = bull_val - bull_power[i-2]  # Rising if positive
            bear_momentum = bear_val - bear_power[i-2]  # Falling if negative
        else:
            bull_momentum = 0
            bear_momentum = 0
        
        if position == 0:
            # Long: Bull Power > 0 and rising + price > 1d EMA50 + volume confirm
            long_condition = (bull_val > 0) and (bull_momentum > 0) and (price > ema_50_1d_val) and volume_confirm
            # Short: Bear Power < 0 and falling + price < 1d EMA50 + volume confirm
            short_condition = (bear_val < 0) and (bear_momentum < 0) and (price < ema_50_1d_val) and volume_confirm
            
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
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Regime exit: price below 1d EMA50
                elif price < ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Elder Ray exit: Bull Power turns negative
                elif bull_val <= 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Regime exit: price above 1d EMA50
                elif price > ema_50_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Elder Ray exit: Bear Power turns positive
                elif bear_val >= 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0