#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Donchian channel breakout for direction and 1h volume spike for entry timing.
Long when price breaks above 4h Donchian upper (20) AND 1h volume > 2.0x 20-period average volume.
Short when price breaks below 4h Donchian lower (20) AND 1h volume > 2.0x 20-period average volume.
Exit when price returns to 4h Donchian midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.20) targeting ~20-50 trades/year on 1h timeframe.
Combines 4h price structure (Donchian channel) for regime/trend and 1h volume confirmation for precise entry.
Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper/lower on 4h
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # 1h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    start_idx = max(20, 20)  # Donchian20 and volMA20 need 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        mid = donchian_mid_aligned[i]
        
        if position == 0:
            # Long: Break above 4h Donchian upper AND volume spike
            if close[i] > upper and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: Break below 4h Donchian lower AND volume spike
            elif close[i] < lower and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price returns to 4h Donchian midpoint
            if position == 1 and close[i] <= mid:
                exit_signal = True
            elif position == -1 and close[i] >= mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_Breakout_VolumeSpike_4hDirection_SessionFilter_MidExit_ATRTrailingStop"
timeframe = "1h"
leverage = 1.0