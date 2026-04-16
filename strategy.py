#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (R1/S1) from daily pivots + 12h volume confirmation + 1d ADX trend filter
# Long when price breaks above R1 AND volume > 1.2x 12h average volume AND 1d ADX > 25
# Short when price breaks below S1 AND volume > 1.2x 12h average volume AND 1d ADX > 25
# ATR trailing stop (2.0x ATR) to manage risk
# Camarilla levels provide clear support/resistance, volume confirms conviction, ADX filters for trending markets
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Calculate Camarilla levels
    camarilla_r1 = pivot + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = pivot - 1.1 * (high_1d - low_1d) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Volume Confirmation (average volume) ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values  # 20 periods average
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === 1d ADX trend filter (14-period) ===
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d_arr - low_1d_arr
    tr2 = np.abs(high_1d_arr - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d_arr - np.roll(close_1d_arr, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d_arr - np.roll(high_1d_arr, 1)) > (np.roll(low_1d_arr, 1) - low_1d_arr), 
                       np.maximum(high_1d_arr - np.roll(high_1d_arr, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d_arr, 1) - low_1d_arr) > (high_1d_arr - np.roll(high_1d_arr, 1)), 
                        np.maximum(np.roll(low_1d_arr, 1) - low_1d_arr, 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h ATR for trailing stop (14-period) ===
    high_12h_arr = df_12h['high'].values
    low_12h_arr = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    tr1_12h = high_12h_arr - low_12h_arr
    tr2_12h = np.abs(high_12h_arr - np.roll(close_12h_arr, 1))
    tr3_12h = np.abs(low_12h_arr - np.roll(close_12h_arr, 1))
    tr2_12h[0] = tr1_12h[0]
    tr3_12h[0] = tr1_12h[0]
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
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
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma_12h_aligned[i]
        adx_val = adx_aligned[i]
        atr_val = atr_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.2x 12h average volume
        vol_confirm = volume[i] > vol_ma_val * 1.2
        
        # ADX filter: trending market (ADX > 25)
        trend_filter = adx_val > 25
        
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
            # Long when: price breaks above R1 AND volume confirmation AND trend filter
            if price > r1_val and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below S1 AND volume confirmation AND trend filter
            elif price < s1_val and vol_confirm and trend_filter:
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

name = "12h_Camarilla_R1_S1_1dADX25_12hVolume1.2x_ATRTrail_2.0x"
timeframe = "12h"
leverage = 1.0