#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) + 1d volume spike + 12h ADX trend filter
# Long when price breaks above Donchian high(20) AND volume > 1.5x 1d average volume AND 12h ADX > 25
# Short when price breaks below Donchian low(20) AND volume > 1.5x 1d average volume AND 12h ADX > 25
# ATR trailing stop (2.5x ATR) to manage risk
# Donchian provides clear breakout levels, volume confirms conviction, ADX filters for trending markets
# Target: 50-150 total trades over 4 years (12-38/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === 1d Volume Confirmation (average volume) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 12h ADX trend filter (14-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[np.isnan(dx)] = 0
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 4h ATR for trailing stop (14-period) ===
    high_4h_arr = df_4h['high'].values
    low_4h_arr = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    tr1_4h = high_4h_arr - low_4h_arr
    tr2_4h = np.abs(high_4h_arr - np.roll(close_4h_arr, 1))
    tr3_4h = np.abs(low_4h_arr - np.roll(close_4h_arr, 1))
    tr2_4h[0] = tr1_4h[0]
    tr3_4h[0] = tr1_4h[0]
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        adx_val = adx_aligned[i]
        atr_val = atr_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # ADX filter: trending market (ADX > 25)
        trend_filter = adx_val > 25
        
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
            # Long when: price breaks above Donchian high AND volume confirmation AND trend filter
            if price > upper and vol_confirm and trend_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Donchian low AND volume confirmation AND trend filter
            elif price < lower and vol_confirm and trend_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dVolume1.5x_12hADX25_ATRTrail_2.5x"
timeframe = "4h"
leverage = 1.0