#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 12h volume confirmation and 1d EMA34 trend filter
# Entry on breakout above Camarilla R1 (long) or below S1 (short) on 6h timeframe.
# 12h volume > 1.5x 20-period average confirms institutional participation.
# 1d EMA34 acts as trend filter: only long when price > EMA34, short when price < EMA34.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Camarilla levels derived from prior 12h session; breakout indicates momentum with volume confirmation.
# Works in both bull and bear markets via trend filter and volume-based entry validation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data for Camarilla calculation ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 12h Volume Confirmation (20-period average) ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 1d EMA34 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Camarilla Levels from prior 12h session ===
    # Calculate from previous completed 12h bar (HLC of prior bar)
    # We shift by 1 to use prior bar's HLC for current bar's levels
    high_12h_shift = np.roll(df_12h['high'].values, 1)
    low_12h_shift = np.roll(df_12h['low'].values, 1)
    close_12h_shift = np.roll(df_12h['close'].values, 1)
    high_12h_shift[0] = np.nan  # First bar has no prior
    low_12h_shift[0] = np.nan
    close_12h_shift[0] = np.nan
    
    # Camarilla R1, S1 calculation
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_12h_shift - low_12h_shift
    r1 = close_12h_shift + 1.1 * camarilla_range / 12
    s1 = close_12h_shift - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema34_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34_val = ema34_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        
        # === EXIT LOGIC (trend filter reversal) ===
        if position == 1:  # Long position
            # Exit when price crosses below 1d EMA34
            if price < ema34_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 1d EMA34
            if price > ema34_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above R1 AND price > EMA34 AND volume confirmation
            if price > r1_val and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below S1 AND price < EMA34 AND volume confirmation
            elif price < s1_val and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R1S1_12hVolumeConfirm_1dEMA34"
timeframe = "6h"
leverage = 1.0