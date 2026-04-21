#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLine_12hTrend_Regime_v1
Hypothesis: Elder Ray (Bull/Bear Power) crossing zero line with 12h EMA50 trend filter on 6h timeframe.
Works in bull markets via Bull Power > 0 + price above 12h EMA50 for longs.
Works in bear markets via Bear Power < 0 + price below 12h EMA50 for shorts.
Adds volume confirmation (>1.5x 20-bar average) to avoid false signals.
Discrete sizing (0.25) and ATR-based stop (2.0x) to manage drawdown.
Target: 50-150 total trades over 4 years by requiring confluence of Elder Ray zero cross, trend, and volume.
Uses 12h EMA for HTF trend alignment to reduce whipsaws.
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
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
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
        bp = bull_power[i]
        br = bear_power[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime from 12h EMA50
        is_bull = price > ema_50_12h_val
        is_bear = price < ema_50_12h_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long when Bull Power crosses above zero
                long_condition = (bp > 0) and (bp <= ema_13[i] * 0.001 + 1e-9) and vol_conf  # just crossed above zero
                short_condition = False
            else:  # bear regime
                # Bear regime: short when Bear Power crosses below zero
                short_condition = (br < 0) and (br >= -ema_13[i] * 0.001 - 1e-9) and vol_conf  # just crossed below zero
                long_condition = False
            
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
                # Exit if Bull Power turns negative (momentum lost)
                elif bp < 0:
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
                # Exit if Bear Power turns positive (momentum lost)
                elif br > 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ZeroLine_12hTrend_Regime_v1"
timeframe = "6h"
leverage = 1.0