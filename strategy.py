#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly Donchian(20) breakout with 1d volume confirmation and ATR stoploss.
Long when price breaks above weekly Donchian high AND daily volume > 1.5x 20-day average.
Short when price breaks below weekly Donchian low AND daily volume > 1.5x 20-day average.
Exit when price retouches weekly Donchian midpoint or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to balance return and drawdown control.
Designed for 1d timeframe to target 10-25 trades/year per symbol (40-100 total over 4 years).
Works in both bull and bear markets by trading breakouts with volume confirmation to filter false signals.
Weekly Donchian provides major structural support/resistance; volume confirmation ensures institutional participation.
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
    
    # Calculate weekly Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        midpoint = donchian_mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high AND volume spike
            if (price > upper and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below weekly Donchian low AND volume spike
            elif (price < lower and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches weekly Donchian midpoint
            if position == 1 and price <= midpoint:
                exit_signal = True
            elif position == -1 and price >= midpoint:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyDonchian20_Breakout_VolumeConfirmation_ATRStop"
timeframe = "1d"
leverage = 1.0