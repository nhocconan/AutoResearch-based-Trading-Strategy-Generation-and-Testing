#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1w EMA50 trend filter
# Long when price > Alligator Jaw AND price > 1w EMA50 AND volume > 1.5x 1d average volume
# Short when price < Alligator Jaw AND price < 1w EMA50 AND volume > 1.5x 1d average volume
# ATR trailing stop (2.0x ATR) to manage risk
# Williams Alligator uses smoothed moving averages (Jaw=13, Teeth=8, Lips=5) to identify trends
# Jaw (blue line) acts as dynamic support/resistance
# EMA50 filter ensures alignment with long-term weekly trend
# Volume confirmation adds conviction to breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # === 12h Williams Alligator (Jaw: 13-period SMMA) ===
    # Williams Alligator uses Smoothed Moving Average (SMMA)
    def smma(source, length):
        # SMMA is similar to EMA but with alpha = 1/length
        # First value is SMA, then recursive smoothing
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < length:
            return result
        # Initial SMA
        result[length-1] = np.mean(source[:length])
        # Subsequent SMMA values
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)  # Jaw: 13-period SMMA (blue line)
    teeth = smma(close_12h, 8)  # Teeth: 8-period SMMA (red line)
    lips = smma(close_12h, 5)   # Lips: 5-period SMMA (green line)
    
    # Align Jaw to 12h timeframe (we use Jaw as the main trend indicator)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    
    # === 1d Volume Confirmation ===
    # Calculate 1d average volume (24 periods of 12h data = 1 day)
    vol_ma_1d = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # === 12h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_1d[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume for confirmation
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
            # Long when: price > Jaw AND price > 1w EMA50 AND volume confirmation
            if price > jaw_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price < Jaw AND price < 1w EMA50 AND volume confirmation
            elif price < jaw_val and price < ema_val and vol_confirm:
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

name = "12h_WilliamsAlligator_1wEMA50_Volume1.5x_ATRTrail_2.0x"
timeframe = "12h"
leverage = 1.0