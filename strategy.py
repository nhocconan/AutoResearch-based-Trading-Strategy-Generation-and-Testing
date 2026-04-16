#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d EMA13 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA13 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power < 0 AND price < 1d EMA13 AND volume > 1.5x average
# ATR trailing stop (2.0x ATR) to manage risk
# Works in bull/bear: Elder Ray shows power balance, EMA13 filters trend, volume confirms conviction
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA13 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # === 6th EMA13 for Elder Ray calculation ===
    ema_13_6th = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).values
    
    # Elder Ray components
    bull_power = high - ema_13_6th  # High - EMA13
    bear_power = low - ema_13_6th   # Low - EMA13
    
    # === 6th Volume Confirmation ===
    vol_ma_6th = pd.Series(volume).rolling(window=4, min_periods=4).mean().values  # 4 periods of 6h = 1 day
    
    # === 6th ATR for trailing stop (10-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 20
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ma_6th[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_13_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_confirm = volume[i] > vol_ma_6th[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Bull Power > 0 AND Bear Power < 0 AND price > EMA13 AND volume confirmation
            if bull > 0 and bear < 0 and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: Bear Power < 0 AND Bull Power < 0 AND price < EMA13 AND volume confirmation
            elif bear < 0 and bull < 0 and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA13_Volume1.5x_ATRTrail_2.0x"
timeframe = "6h"
leverage = 1.0