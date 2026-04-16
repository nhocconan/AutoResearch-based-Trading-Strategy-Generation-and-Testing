#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# Long when Lips > Teeth > Jaw AND price > 1d EMA50 AND volume > 1.5x 20-period average volume
# Short when Lips < Teeth < Jaw AND price < 1d EMA50 AND volume > 1.5x 20-period average volume
# ATR trailing stop (2.0x ATR) to manage risk
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 12h timeframe
# Williams Alligator identifies trend direction and alignment, 1d EMA50 filter avoids counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Williams Alligator (SMMA) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(values, period):
        sma = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            sma[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                sma[i] = (sma[i-1] * (period-1) + values[i]) / period
        return sma
    
    jaw = smma(close_12h, 13)  # Jaw: 13-period SMMA
    teeth = smma(close_12h, 8)  # Teeth: 8-period SMMA
    lips = smma(close_12h, 5)   # Lips: 5-period SMMA
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === 1d EMA50 filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 12h ATR for trailing stop (14-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_for_atr = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h_for_atr, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h_for_atr, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
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
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr_aligned[i]
        
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
            # Alligator alignment: Lips > Teeth > Jaw (bullish)
            bullish_alignment = lips_val > teeth_val and teeth_val > jaw_val
            # Alligator alignment: Lips < Teeth < Jaw (bearish)
            bearish_alignment = lips_val < teeth_val and teeth_val < jaw_val
            
            # Long when: Bullish alignment AND price > EMA50 AND volume confirmation
            if bullish_alignment and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: Bearish alignment AND price < EMA50 AND volume confirmation
            elif bearish_alignment and price < ema_val and vol_confirm:
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

name = "12h_WilliamsAlligator_1dEMA50_Volume1.5x_ATRTrail"
timeframe = "12h"
leverage = 1.0