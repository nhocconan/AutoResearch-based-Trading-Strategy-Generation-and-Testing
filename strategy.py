#!/usr/bin/env python3
"""
Hypothesis: 1h 4-hour Donchian channel breakout with 1d volume confirmation and session filter.
Long when price breaks above 4h Donchian upper (20) AND 1d volume > 1.5x 20-day average volume AND hour in [08,20) UTC.
Short when price breaks below 4h Donchian lower (20) AND 1d volume > 1.5x 20-day average volume AND hour in [08,20) UTC.
Exit when price returns to 4h Donchian middle (midpoint of upper/lower) OR ATR trailing stop (1.5*ATR from extreme).
Uses 4h for structure/direction, 1h only for entry timing precision. Session filter reduces noise trades.
Target: ~20-40 trades/year on 1h timeframe with discrete sizing 0.20 to minimize fee drag.
Works in bull markets (breakouts up) and bear markets (breakouts down) by capturing expansion phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours < 20)
    
    # Calculate 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper/lower: 20-period high/low
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Align to 1h timeframe (already delayed for completed 4h bar)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    
    # Calculate 1d volume average (20-period) for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR(14) for 1h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian20, vol_ma20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        atr_val = atr[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper AND volume spike AND in session
            if price > donch_high_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: Price breaks below 4h Donchian lower AND volume spike AND in session
            elif price < donch_low_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price returns to 4h Donchian middle (mean reversion)
            if position == 1 and price < donch_mid_val:
                exit_signal = True
            elif position == -1 and price > donch_mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 1.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 1.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hDonchian_Breakout_1dVolumeSpike_SessionFilter_MidExit_ATRTrailingStop"
timeframe = "1h"
leverage = 1.0