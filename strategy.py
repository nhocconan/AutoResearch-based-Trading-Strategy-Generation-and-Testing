#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d ADX25 trend filter + volume confirmation + ATR trailing stop
# Entry on breakout above Donchian upper (long) or below lower (short) on 6h timeframe.
# 1d ADX > 25 acts as trend filter: only trade in trending markets (avoid chop/range).
# Volume confirmation: current 6h volume > 1.5x 20-period average of 6h volume.
# ATR-based trailing stop (2.0x ATR) to manage risk and reduce whipsaws.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in both bull and bear markets via trend filter and volatility-based stops.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for ADX (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 6h Donchian Channel (20-period) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Donchian upper/lower = rolling max(high,20) / min(low,20)
    donch_upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_6h, donch_upper_6h)
    donch_lower_aligned = align_htf_to_ltf(prices, df_6h, donch_lower_6h)
    
    # === 1d ADX (trend filter) ===
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM = max(high - high_prev, 0) if > low_prev - low else 0
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_period = 14
    tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=tr_period, min_periods=tr_period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=tr_period, min_periods=tr_period).sum().values
    
    # +DI, -DI, DX
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    # ADX = smoothed DX
    adx_1d = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Volume Confirmation (20-period average) ===
    volume_6h = df_6h['volume'].values
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # === 6h ATR for trailing stop (14-period) ===
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr2_6h[0] = tr1_6h[0]
    tr3_6h[0] = tr1_6h[0]
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
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
        if (np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        donch_upper = donch_upper_aligned[i]
        donch_lower = donch_lower_aligned[i]
        adx_val = adx_aligned[i]
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
            # Exit when 1d ADX drops below 20 (trend weakening)
            if adx_val < 20:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when 1d ADX drops below 20 (trend weakening)
            if adx_val < 20:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian upper AND ADX > 25 AND volume confirmation
            if price > donch_upper and adx_val > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Donchian lower AND ADX > 25 AND volume confirmation
            elif price < donch_lower and adx_val > 25 and vol_confirm:
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

name = "6h_Donchian20_1dADX25_VolumeConfirm_ATRTrail"
timeframe = "6h"
leverage = 1.0