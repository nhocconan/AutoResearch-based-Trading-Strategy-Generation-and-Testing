#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA34 trend filter, volume confirmation, and ATR trailing stop.
# Uses 4h for signal direction (trend + structure) and 1h for precise entry timing.
# Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years.
# Works in bull/bear via trend filter and volatility-based stops.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) once
    hours = prices.index.hour
    
    # === 1h data for Donchian, volume ===
    df_1h = prices  # primary timeframe is 1h
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # === Donchian Channel (20-period) on 1h ===
    highest_20 = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    
    # === 1h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # === 1h ATR for trailing stop (14-period) ===
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h EMA34 (trend filter) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Skip if any data is NaN
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(ema34_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_1h[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        price = close[i]
        upper_val = highest_20[i]
        lower_val = lowest_20[i]
        ema34_val = ema34_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume
        atr_val = atr_1h[i]
        
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
                lowest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (trend filter reversal) ===
        if position == 1:  # Long position
            # Exit when price crosses below 4h EMA34
            if price < ema34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 4h EMA34
            if price > ema34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above upper band AND price > EMA34 AND volume confirmation
            if price > upper_val and price > ema34_val and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
                highest_since_entry = price
                lowest_since_entry = price
                continue
            # Short when: price breaks below lower band AND price < EMA34 AND volume confirmation
            elif price < lower_val and price < ema34_val and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
                highest_since_entry = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_4hEMA34_VolumeConfirm_ATRTrail_Session"
timeframe = "1h"
leverage = 1.0