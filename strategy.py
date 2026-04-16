#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 12h Donchian(20) breakout with volume confirmation
# Long when: 4h Choppiness > 61.8 (range) AND price breaks above 12h Donchian(20) high AND volume > 1.5x 20-period average volume
# Short when: 4h Choppiness > 61.8 (range) AND price breaks below 12h Donchian(20) low AND volume > 1.5x 20-period average volume
# Exit when: 4h Choppiness < 38.2 (trending regime) OR opposite breakout occurs
# ATR trailing stop (2.0x ATR) for risk management
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag on 4h timeframe
# Choppiness filter avoids whipsaws in strong trends, focusing on mean-reversion in ranging markets
# Donchian breakouts capture momentum within range, volume confirmation adds conviction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Choppiness Index (regime filter) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_4h).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index = 100 * log10(sum(ATR14) / range14) / log10(14)
    # Avoid division by zero and log of zero
    choppiness = np.zeros_like(close_4h)
    mask = (range_14 > 0) & (sum_atr_14 > 0)
    choppiness[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    choppiness[~mask] = 50.0  # neutral when invalid
    
    choppiness_aligned = align_htf_to_ltf(prices, df_4h, choppiness)
    
    # === 12h Donchian(20) channels ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 12h Volume Spike Confirmation (20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 4h ATR for trailing stop (14-period) ===
    # Use already calculated atr_4h from Choppiness calculation
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
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
        if (np.isnan(choppiness_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop_value = choppiness_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume for spike
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
        
        # === EXIT LOGIC: Choppiness < 38.2 (trending regime) ===
        if position != 0 and chop_value < 38.2:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat or reversing) ===
        if position == 0:
            # Long when: Choppiness > 61.8 (range) AND price breaks above Donchian(20) high AND volume spike
            if chop_value > 61.8 and price > donchian_high_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: Choppiness > 61.8 (range) AND price breaks below Donchian(20) low AND volume spike
            elif chop_value > 61.8 and price < donchian_low_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        # Reverse position if opposite signal in ranging market
        elif position == 1 and chop_value > 61.8 and price < donchian_low_val and vol_confirm:
            signals[i] = -0.25
            position = -1
            entry_price = price
            lowest_since_entry = price
            highest_since_entry = 0.0
            continue
        elif position == -1 and chop_value > 61.8 and price > donchian_high_val and vol_confirm:
            signals[i] = 0.25
            position = 1
            entry_price = price
            highest_since_entry = price
            lowest_since_entry = 0.0
            continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_ChoppinessRange_12hDonchian20_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0