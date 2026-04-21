#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrendRegime_v1
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) on 6h with 1-week EMA34 trend filter captures institutional buying/selling pressure. Long when Bull Power > 0 and Bear Power improving (less negative) in bull weekly trend; Short when Bear Power < 0 and Bull Power worsening (less positive) in bear weekly trend. Uses discrete sizing (0.25) and ATR-based stop (2.0x) to minimize fee drag. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 6h EMA13 for Elder Ray calculation ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Elder Ray: Bull Power and Bear Power ===
    bull_power = high - ema_13  # Buying power
    bear_power = ema_13 - low   # Selling power
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        ema_13_val = ema_13[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Trend regime from weekly EMA34
        is_bull_week = price > ema_34_1w_val
        is_bear_week = price < ema_34_1w_val
        
        if position == 0:
            # Long conditions: Bullish weekly + buying pressure improving
            if is_bull_week:
                long_condition = (bull > 0) and (bull > bull_power[i-1])  # Bull Power > 0 and rising
                # Short only if strong selling pressure in bull trend (unlikely but possible)
                short_condition = (bear > 0) and (bear > bear_power[i-1]) and (price < ema_13_val * 0.98)
            # Short conditions: Bearish weekly + selling pressure improving
            else:  # is_bear_week
                short_condition = (bear > 0) and (bear > bear_power[i-1])  # Bear Power > 0 and rising
                # Long only if strong buying pressure in bear trend (unlikely but possible)
                long_condition = (bull > 0) and (bull > bull_power[i-1]) and (price > ema_13_val * 1.02)
            
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
                # Exit if Elder Ray turns negative (buying pressure gone)
                elif bull_power[i] <= 0:
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
                # Exit if Elder Ray turns positive (selling pressure gone)
                elif bear_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrendRegime_v1"
timeframe = "6h"
leverage = 1.0