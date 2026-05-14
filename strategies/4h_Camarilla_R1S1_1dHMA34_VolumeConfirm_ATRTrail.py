#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d HMA(34) trend filter and volume confirmation
# Long when price breaks above Camarilla R1 level with price > 1d HMA34 and volume > 1.5x 20-period average
# Short when price breaks below Camarilla S1 level with price < 1d HMA34 and volume > 1.5x 20-period average
# ATR-based trailing stop (2.0x ATR) to manage risk and reduce whipsaws
# Camarilla pivot levels derived from previous 1d OHLC, providing intraday support/resistance
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag
# Works in both bull and bear markets via trend filter and volatility-based stops

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Camarilla Pivot Levels (from previous 1d OHLC) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    camarilla_r1 = close_1d + camarilla_range
    camarilla_s1 = close_1d - camarilla_range
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 4h Volume Confirmation (20-period average) ===
    df_4h = get_htf_data(prices, '4h')
    vol_ma_20 = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # === 4h ATR for trailing stop (14-period) ===
    tr1 = df_4h['high'].values - df_4h['low'].values
    tr2 = np.abs(df_4h['high'].values - np.roll(df_4h['close'].values, 1))
    tr3 = np.abs(df_4h['low'].values - np.roll(df_4h['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # === 1d HMA34 (trend filter) ===
    half_len = 34 // 2
    sqrt_len = int(np.sqrt(34))
    wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(1, half_len+1)) / np.sum(np.arange(1, half_len+1)), raw=True
    ).values
    wma_full = pd.Series(close_1d).rolling(window=34, min_periods=34).apply(
        lambda x: np.dot(x, np.arange(1, 35)) / np.sum(np.arange(1, 35)), raw=True
    ).values
    wma_diff = 2 * wma_half - wma_full
    hma_34 = pd.Series(wma_diff).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.sum(np.arange(1, sqrt_len+1)), raw=True
    ).values
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_34)
    
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
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(hma_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
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
            # Exit when price crosses below 1d HMA34
            if price < hma_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 1d HMA34
            if price > hma_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Camarilla R1 AND price > HMA34 AND volume confirmation
            if price > r1_val and price > hma_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Camarilla S1 AND price < HMA34 AND volume confirmation
            elif price < s1_val and price < hma_val and vol_confirm:
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

name = "4h_Camarilla_R1S1_1dHMA34_VolumeConfirm_ATRTrail"
timeframe = "4h"
leverage = 1.0