#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 (1d) AND price > weekly EMA50 AND volume > 1.5x 20-period average volume
# Short when price breaks below Camarilla S1 (1d) AND price < weekly EMA50 AND volume > 1.5x 20-period average volume
# ATR trailing stop (2.0x ATR) to manage risk
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 12h timeframe
# Camarilla pivot levels from 1d provide precise intraday support/resistance, weekly EMA50 filters trend direction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly EMA50 (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily Camarilla Pivot Levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Volume Spike Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for spike
        atr_val = atr_12h[i]
        
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
            # Long when: price breaks above Camarilla R1 AND price > weekly EMA50 AND volume spike
            if price > r1_val and price > ema_50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Camarilla S1 AND price < weekly EMA50 AND volume spike
            elif price < s1_val and price < ema_50_val and vol_confirm:
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

name = "12h_Camarilla_R1S1_1wEMA50_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0