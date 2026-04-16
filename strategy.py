#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ATR breakout with 1w trend filter and volume confirmation
# Long when price closes above previous 12h high + 1.5*ATR(14) AND price > 1w EMA50 AND volume > 1.5x 12h average volume
# Short when price closes below previous 12h low - 1.5*ATR(14) AND price < 1w EMA50 AND volume > 1.5x 12h average volume
# ATR trailing stop (2.0x ATR) to manage risk
# Weekly EMA50 ensures alignment with major trend
# ATR-based breakout filters noise, volume adds conviction
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # === 12h ATR(14) for breakout and stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 12h high/low for breakout levels ===
    high_12h = pd.Series(high).rolling(window=2, min_periods=2).max().values  # 2 periods of 6h = 12h
    low_12h = pd.Series(low).rolling(window=2, min_periods=2).min().values
    
    # Align 12h high/low to 6h timeframe (each 12h bar = 2x 6h bars)
    high_12h_shifted = np.roll(high_12h, 1)
    low_12h_shifted = np.roll(low_12h, 1)
    high_12h_shifted[0] = np.nan
    low_12h_shifted[0] = np.nan
    
    # === 12h Volume Confirmation ===
    vol_ma_12h = pd.Series(volume).rolling(window=2, min_periods=2).mean().values  # 2 periods of 6h = 12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(high_12h_shifted[i]) or
            np.isnan(low_12h_shifted[i]) or
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_aligned[i]
        high_break = high_12h_shifted[i] + 1.5 * atr[i]
        low_break = low_12h_shifted[i] - 1.5 * atr[i]
        vol_confirm = volume[i] > vol_ma_12h[i] * 1.5
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
            # Long when: price closes above 12h high + 1.5*ATR AND price > EMA50 AND volume confirmation
            if price > high_break and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price closes below 12h low - 1.5*ATR AND price < EMA50 AND volume confirmation
            elif price < low_break and price < ema_val and vol_confirm:
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

name = "12h_ATRBreakout_1wEMA50_Volume1.5x_ATRTrail_2.0x"
timeframe = "12h"
leverage = 1.0