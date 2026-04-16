#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly pivot points (R2/S2) with volume confirmation and weekly ADX trend filter
# Long when price crosses above R2 AND volume > 1.3x weekly average volume AND weekly ADX > 20
# Short when price crosses below S2 AND volume > 1.3x weekly average volume AND weekly ADX > 20
# ATR trailing stop (2.5x ATR) to manage risk
# Weekly pivots provide strong weekly support/resistance, volume confirms conviction, ADX filters for trending markets
# Target: 50-100 total trades over 4 years (12-25/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Weekly Pivot Points (R2, S2) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point
    pivot = (high_1w + low_1w + close_1w) / 3
    # Calculate Weekly pivot levels (R2, S2)
    weekly_r2 = pivot + (high_1w - low_1w)
    weekly_s2 = pivot - (high_1w - low_1w)
    
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # === 1d Weekly Volume Confirmation (average volume) ===
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values  # 10 weeks average
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === 1d Weekly ADX trend filter (14-period) ===
    high_1w_arr = df_1w['high'].values
    low_1w_arr = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # True Range
    tr1 = high_1w_arr - low_1w_arr
    tr2 = np.abs(high_1w_arr - np.roll(close_1w_arr, 1))
    tr3 = np.abs(low_1w_arr - np.roll(close_1w_arr, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w_arr - np.roll(high_1w_arr, 1)) > (np.roll(low_1w_arr, 1) - low_1w_arr), 
                       np.maximum(high_1w_arr - np.roll(high_1w_arr, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w_arr, 1) - low_1w_arr) > (high_1w_arr - np.roll(high_1w_arr, 1)), 
                        np.maximum(np.roll(low_1w_arr, 1) - low_1w_arr, 0), 0)
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
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === 1d Weekly ATR for trailing stop (14-period) ===
    high_1w_arr = df_1w['high'].values
    low_1w_arr = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    tr1_1w = high_1w_arr - low_1w_arr
    tr2_1w = np.abs(high_1w_arr - np.roll(close_1w_arr, 1))
    tr3_1w = np.abs(low_1w_arr - np.roll(close_1w_arr, 1))
    tr2_1w[0] = tr1_1w[0]
    tr3_1w[0] = tr1_1w[0]
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
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
        if (np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or
            np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r2_val = weekly_r2_aligned[i]
        s2_val = weekly_s2_aligned[i]
        vol_ma_val = vol_ma_1w_aligned[i]
        adx_val = adx_aligned[i]
        atr_val = atr_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x weekly average volume
        vol_confirm = volume[i] > vol_ma_val * 1.3
        
        # ADX filter: trending market (ADX > 20)
        trend_filter = adx_val > 20
        
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
            # Long when: price crosses above R2 AND volume confirmation AND trend filter
            if price > r2_val and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price crosses below S2 AND volume confirmation AND trend filter
            elif price < s2_val and vol_confirm and trend_filter:
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

name = "1d_WeeklyPivot_R2_S2_1wVolume1.3x_1wADX20_ATRTrail_2.5x"
timeframe = "1d"
leverage = 1.0