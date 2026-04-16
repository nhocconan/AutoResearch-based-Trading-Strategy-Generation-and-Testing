#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
# In choppy markets (CHOP > 61.8): mean reversion at Donchian bands
# In trending markets (CHOP < 38.2): breakout continuation
# Uses daily Choppiness Index to avoid false signals in sideways markets
# Volume confirmation (1.5x 20-period average) adds conviction
# ATR trailing stop (2.5x) manages risk
# Designed for low trade frequency (target: 50-150 total over 4 years) on 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Choppiness Index (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) and min(low) over 14 periods
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr) / (max(hh) - min(ll))) / log10(14)
    # Avoid division by zero
    range_hl = max_hh - min_ll
    chop_raw = np.where(range_hl > 0, sum_tr / range_hl, 1.0)
    chop_raw = np.where(chop_raw > 0, chop_raw, 1.0)
    chop_1d = 100 * np.log10(chop_raw) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)  # neutral if undefined
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Donchian(20) channels ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === 12h Volume Confirmation (1.5x 20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 12h ATR for trailing stop (15-period) ===
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(df_12h['close'].values, 1))
    tr3_12h = np.abs(low_12h - np.roll(df_12h['close'].values, 1))
    tr2_12h[0] = tr1_12h[0]
    tr3_12h[0] = tr1_12h[0]
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=15, min_periods=15).mean().values
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
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop_val = chop_1d_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine market regime from daily Choppiness Index
            is_choppy = chop_val > 61.8  # range-bound market
            is_trending = chop_val < 38.2  # trending market
            
            if is_choppy:
                # In choppy markets: mean reversion at Donchian bands
                # Long when price touches/below lower band AND volume confirmation
                if price <= donchian_low_val and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when price touches/above upper band AND volume confirmation
                elif price >= donchian_high_val and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
            elif is_trending:
                # In trending markets: breakout continuation
                # Long when price breaks above Donchian high AND volume confirmation
                if price > donchian_high_val and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when price breaks below Donchian low AND volume confirmation
                elif price < donchian_low_val and vol_confirm:
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

name = "12h_ChopRegime_Donchian20_Volume1.5x_ATRTrail"
timeframe = "12h"
leverage = 1.0