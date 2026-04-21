#!/usr/bin/env python3
"""
6h_ElderRay_Power_Regime_Breakout
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price > 1d EMA34, volume > 1.5x MA.
Short when Bear Power < 0 and falling, Bull Power < 0 and rising, price < 1d EMA34, volume > 1.5x MA.
ATR trailing stop (2.5x ATR) manages risk. Works in bull via Elder Ray strength, in bear via reversals from extreme Power readings.
Position size 0.25 balances risk/return. Target ~12-37 trades/year per symbol (50-150 total over 4 years).
Uses 6h primary timeframe with 1d HTF for trend alignment, avoiding overtrading while capturing multi-day moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power and Bear Power
    bull_power = high_6h - ema_13  # High - EMA13
    bear_power = low_6h - ema_13   # Low - EMA13
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high_6h[1:] - low_6h[1:]
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Elder Ray momentum: rising Bull Power or falling Bear Power
        bull_power_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bear_power_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        bull_power_falling = bull_power[i] < bull_power[i-1] if i > 0 else False
        bear_power_rising = bear_power[i] > bear_power[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, price > 1d EMA34, volume confirmation
            if bull_power[i] > 0 and bull_power_rising and bear_power[i] < 0 and price > ema_34_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Bear Power < 0 and falling, Bull Power < 0, price < 1d EMA34, volume confirmation
            elif bear_power[i] < 0 and bear_power_falling and bull_power[i] < 0 and price < ema_34_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_Regime_Breakout"
timeframe = "6h"
leverage = 1.0