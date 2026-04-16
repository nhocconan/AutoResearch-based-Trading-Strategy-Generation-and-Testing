#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA13 trend filter and volume confirmation
# Elder Ray = Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power > previous Bear Power (bullish momentum) AND price > 1d EMA13 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum) AND price < 1d EMA13 AND volume > 1.5x average
# ATR trailing stop (2.0x ATR) to manage risk
# Elder Ray captures institutional buying/selling pressure; EMA13 filter ensures trend alignment; volume confirms conviction
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag

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
    
    # === Elder Ray (Bull/Bear Power) ===
    # Calculate EMA13 for high and low prices
    ema13_high = pd.Series(high).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(low).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13_high  # High - EMA13
    bear_power = low - ema13_low    # Low - EMA13
    
    # === 1d Volume Confirmation ===
    vol_ma_1d = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 periods of 6h = 4d? Wait, 6h * 4 = 24h = 1d
    
    # === 6h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
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
            np.isnan(vol_ma_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_13_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr[i]
        
        # Previous values for momentum
        prev_bull = bull_power[i-1] if i > 0 else 0
        prev_bear = bear_power[i-1] if i > 0 else 0
        
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
            # Long when: Bull Power > 0 AND Bull Power > previous Bull Power (bullish momentum) AND price > EMA13 AND volume confirmation
            if bull_val > 0 and bull_val > prev_bull and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: Bear Power < 0 AND Bear Power < previous Bear Power (bearish momentum) AND price < EMA13 AND volume confirmation
            elif bear_val < 0 and bear_val < prev_bear and price < ema_val and vol_confirm:
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