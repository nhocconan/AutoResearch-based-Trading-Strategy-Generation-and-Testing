#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper channel (20-period high) with price > 1d HMA21 and volume > 1.5x 20-period average
# Short when price breaks below 12h Donchian lower channel (20-period low) with price < 1d HMA21 and volume > 1.5x 20-period average
# ATR-based trailing stop (2.0x ATR) to manage risk and reduce whipsaws
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Works in both bull and bear markets via trend filter and volatility-based stops
# Uses 12h primary timeframe with 1d HTF for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian(20) channels ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper: 20-period high
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: 20-period low
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 12h ATR for trailing stop (14-period) ===
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(df_12h['close'].values, 1))
    tr3 = np.abs(low_12h - np.roll(df_12h['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 1d HMA21 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(1, half_len+1)) / np.sum(np.arange(1, half_len+1)), raw=True
    ).values
    wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).apply(
        lambda x: np.dot(x, np.arange(1, 22)) / np.sum(np.arange(1, 22)), raw=True
    ).values
    wma_diff = 2 * wma_half - wma_full
    hma_21 = pd.Series(wma_diff).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.sum(np.arange(1, sqrt_len+1)), raw=True
    ).values
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
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
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(hma_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        hma_val = hma_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
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
        
        # === EXIT LOGIC (trend filter reversal) ===
        if position == 1:  # Long position
            # Exit when price crosses below 1d HMA21
            if price < hma_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 1d HMA21
            if price > hma_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian high AND price > HMA21 AND volume confirmation
            if price > donch_high_val and price > hma_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Donchian low AND price < HMA21 AND volume confirmation
            elif price < donch_low_val and price < hma_val and vol_confirm:
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

name = "12h_Donchian20_1dHMA21_VolumeConfirm_ATRTrail"
timeframe = "12h"
leverage = 1.0