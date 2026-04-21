#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLine_12hTrend_Regime_v1
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) zero-line cross with 12h trend regime (price vs EMA50) and volume confirmation (>1.5x 20-bar MA). 
In bull regime (price > 12h EMA50), favor longs when Bull Power crosses above zero; in bear regime (price < 12h EMA50), favor shorts when Bear Power crosses below zero. 
Discrete sizing (0.25) and ATR-based stop (2.0x) reduce churn. Target: 50-150 total trades over 4 years by requiring confluence of Elder Ray signal, trend, and volume.
Designed to work in bull (strong buying pressure) and bear (strong selling pressure) markets via Elder Ray's measurement of bull/bear power relative to EMA13.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for trend regime)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for trend regime ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # measures bull power above EMA13
    bear_power = low - ema13   # measures bear power below EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull_regime = price > ema_50_12h_val
        is_bear_regime = price < ema_50_12h_val
        
        if position == 0:
            if is_bull_regime:
                # Bull regime: long when Bull Power crosses above zero (bulls gaining control)
                long_condition = (bull_val > 0) and (bull_power[i-1] <= 0) and vol_conf
                # Avoid shorts in bull regime unless strong bear power
                short_condition = (bear_val < 0) and (bear_power[i-1] >= 0) and vol_conf and (bull_val < -0.5)  # only if weak bull power
            else:  # bear regime
                # Bear regime: short when Bear Power crosses below zero (bears gaining control)
                short_condition = (bear_val < 0) and (bear_power[i-1] >= 0) and vol_conf
                # Avoid longs in bear regime unless strong bull power
                long_condition = (bull_val > 0) and (bull_power[i-1] <= 0) and vol_conf and (bear_val > 0.5)  # only if weak bear power
            
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
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if Bull Power turns negative (bulls losing control)
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
                # Exit if Bear Power turns positive (bears losing control)
                elif bear_val >= 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ZeroLine_12hTrend_Regime_v1"
timeframe = "6h"
leverage = 1.0