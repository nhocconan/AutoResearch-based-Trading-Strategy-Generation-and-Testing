#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper (20) AND 1d ADX > 25 (trending) AND 4h volume > 1.5x 20-median
# Short when price breaks below Donchian lower (20) AND 1d ADX > 25 (trending) AND 4h volume > 1.5x 20-median
# Exit when price reverts to Donchian midpoint or ATR stop (2.0)
# Uses discrete position size 0.25 to balance capture and fee drag. Target: 75-200 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channels (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper and lower
    donch_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_upper + donch_lower) / 2.0
    
    # Align Donchian levels to 4h timeframe (no extra delay needed as we use completed 4h bars)
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    
    # ATR for stoploss (14-period) on 4h
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Get 1h data once before loop for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # === 1h Indicators: Volume median ===
    volume_1h = df_1h['volume'].values
    vol_median_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).median().values
    vol_median_20_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_median_20_1h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14)  # 1d ADX, 4h Donchian/ATR, 1h volume median
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or np.isnan(donch_mid_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_20_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1h volume (aligned)
        vol_1h_aligned = align_htf_to_ltf(prices, df_1h, volume_1h)
        if np.isnan(vol_1h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1h volume > 1.5x 20-period 1h volume median
        vol_threshold = vol_median_20_1h_aligned[i] * 1.5
        vol_confirm = vol_1h_aligned[i] > vol_threshold
        
        # Trend filter: 1d ADX > 25 (trending market)
        trending = adx_aligned[i] > 25.0
        
        # Price levels
        price = close[i]
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        mid = donch_mid_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price returns to midpoint or ATR stoploss
            if price <= mid or price <= entry_price - 2.0 * atr_aligned[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price returns to midpoint or ATR stoploss
            if price >= mid or price >= entry_price + 2.0 * atr_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and trending:
            # LONG CONDITIONS
            # Price breaks above Donchian upper AND volume confirmation AND trending market
            if price > upper and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower AND volume confirmation AND trending market
            elif price < lower and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Donchian20_1dADX_Volume_Filter_v2"
timeframe = "4h"
leverage = 1.0