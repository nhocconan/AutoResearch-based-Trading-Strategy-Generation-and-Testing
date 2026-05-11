#!/usr/bin/env python3
"""
1h_Constitutional_Aggression
Hypothesis: Aggressive 1h momentum entries aligned with 4h trend and 1d regime, filtered by volume and session. Uses 4h Supertrend for trend direction, 1d ADX for regime filtering, and 1h RSI(2) for precise entry timing. Designed for 60-150 total trades over 4 years to avoid fee drag while capturing momentum bursts in both bull and bear markets.
"""

name = "1h_Constitutional_Aggression"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend (Supertrend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for regime (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-calculate hours for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # --- 4h Supertrend for trend direction ---
    atr_period = 10
    atr_mult = 3.0
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_4h = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (atr_mult * atr_4h)
    lower_band = hl2 - (atr_mult * atr_4h)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_4h, np.nan)
    direction = np.full_like(close_4h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend direction to 1h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # --- 1d ADX for regime (filter ranging markets) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1h RSI(2) for entry timing ---
    close_1h = prices['close'].values
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 1h Volume confirmation ---
    volume_1h = prices['volume'].values
    vol_avg_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for Supertrend, ADX, and RSI
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        if not (8 <= hours[i] <= 20):
            if position != 0:
                # Trail stop: exit if adverse move of 1.5% from entry
                if position == 1 and close_1h[i] <= entry_price * 0.985:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1h[i] >= entry_price * 1.015:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        # Skip if any critical values are NaN
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg_1h[i])):
            if position != 0:
                # Trail stop: exit if adverse move of 1.5% from entry
                if position == 1 and close_1h[i] <= entry_price * 0.985:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1h[i] >= entry_price * 1.015:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        # Conditions
        is_uptrend_4h = supertrend_dir_aligned[i] == 1
        is_downtrend_4h = supertrend_dir_aligned[i] == -1
        is_trending_regime = adx_1d_aligned[i] > 25  # Only trade in trending markets
        rsi_oversold = rsi[i] < 15  # Extreme oversold for long
        rsi_overbought = rsi[i] > 85  # Extreme overbought for short
        vol_confirm = volume_1h[i] > 1.5 * vol_avg_1h[i]  # Volume spike
        
        if position == 0:
            # Look for entries
            if is_trending_regime and vol_confirm:
                if is_uptrend_4h and rsi_oversold:
                    # Long in uptrend on extreme pullback
                    signals[i] = 0.20
                    position = 1
                    entry_price = close_1h[i]
                elif is_downtrend_4h and rsi_overbought:
                    # Short in downtrend on extreme bounce
                    signals[i] = -0.20
                    position = -1
                    entry_price = close_1h[i]
        else:
            # Manage existing position with trailing stop
            if position == 1:
                # Long: trail stop if price drops 1.5% from entry
                if close_1h[i] <= entry_price * 0.985:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short: trail stop if price rises 1.5% from entry
                if close_1h[i] >= entry_price * 1.015:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals