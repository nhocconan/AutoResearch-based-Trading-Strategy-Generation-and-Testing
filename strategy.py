#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR trailing stop.
Long when price breaks above weekly Donchian high AND price > 1d EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below weekly Donchian low AND price < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when price retraces to weekly Donchian midpoint or ATR trailing stop hit (2.0*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to target ~15 trades/year per symbol (60 total over 4 years).
Designed for 1d timeframe targeting low trade frequency to minimize fee drag in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian levels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Donchian levels: based on weekly high/low
    # Upper = max(high over last 20 weeks)
    # Lower = min(low over last 20 weeks)
    # Midpoint = (Upper + Lower) / 2
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    donchian_upper = pd.Series(h_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(l_1w).rolling(window=20, min_periods=20).min().values
    donchian_midpoint = (donchian_upper + donchian_lower) / 2.0
    
    # Align weekly Donchian levels to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_midpoint_aligned = align_htf_to_ltf(prices, df_1w, donchian_midpoint)
    
    # Calculate 1d EMA50 for trend filter
    if len(close) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian needs 20, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_midpoint_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        midpoint_val = donchian_midpoint_aligned[i]
        ema_50_val = ema_50[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper AND price > 1d EMA50 AND volume spike
            if (price > upper_val and price > ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below weekly Donchian lower AND price < 1d EMA50 AND volume spike
            elif (price < lower_val and price < ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to weekly Donchian midpoint
            if position == 1 and price <= midpoint_val:
                exit_signal = True
            elif position == -1 and price >= midpoint_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyDonchian20_1dEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0