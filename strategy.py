#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d ATR breakout + volume confirmation
# Long when: CHOP(14) < 38.2 (trending) AND price > 1d ATR breakout (high + 0.5*ATR) AND volume > 1.5x average
# Short when: CHOP(14) < 38.2 (trending) AND price < 1d ATR breakdown (low - 0.5*ATR) AND volume > 1.5x average
# Uses 1d ATR for breakout levels to capture true volatility breakouts
# Choppiness filter avoids whipsaws in ranging markets
# Volume confirmation adds conviction to breakouts
# ATR trailing stop (2x ATR) manages risk
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR for breakout levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate breakout/breakdown levels from previous day
    breakout_level = high_1d + 0.5 * atr_1d
    breakdown_level = low_1d - 0.5 * atr_1d
    
    # Align levels to 4h timeframe
    breakout_level_aligned = align_htf_to_ltf(prices, df_1d, breakout_level)
    breakdown_level_aligned = align_htf_to_ltf(prices, df_1d, breakdown_level)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h Choppiness Index (14-period) ===
    # CHOP = 100 * log10(sum(TR over n) / (max(high) - min(low))) / log10(n)
    tr4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr4h[0] = high[0] - low[0]
    
    atr_4h = pd.Series(tr4h).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14h = max_high - min_low
    chop = np.full_like(close, 50.0)  # default to neutral
    mask = range_14h > 0
    chop[mask] = 100 * np.log10(pd.Series(tr4h).rolling(window=14, min_periods=14).sum().values[mask] / range_14h[mask]) / np.log10(14)
    
    # === 1d Volume Confirmation (using 4h data approximation) ===
    # 6 periods of 4h = 1 day
    vol_ma_1d = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
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
        if (np.isnan(breakout_level_aligned[i]) or 
            np.isnan(breakdown_level_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        chop_val = chop[i]
        breakout_val = breakout_level_aligned[i]
        breakdown_val = breakdown_level_aligned[i]
        atr_val = atr_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma_1d[i] * 1.5  # 1.5x average volume for confirmation
        
        # Only trade in trending markets (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat and trending) ===
        if position == 0 and is_trending:
            # Long when: price breaks above breakout level AND volume confirmation
            if price > breakout_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below breakdown level AND volume confirmation
            elif price < breakdown_val and vol_confirm:
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

name = "4h_ChopTrend_ATRBreakout_Volume1.5x_ATRTrail_2x"
timeframe = "4h"
leverage = 1.0