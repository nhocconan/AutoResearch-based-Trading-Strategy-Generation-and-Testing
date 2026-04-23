#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly Donchian(20) breakout with volume confirmation and ATR stoploss.
Long when price breaks above weekly Donchian upper band AND volume > 1.8x 20-period average.
Short when price breaks below weekly Donchian lower band AND volume > 1.8x 20-period average.
Exit when price retraces 50% of the breakout range or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 1d timeframe to target 7-25 trades/year per symbol (30-100 total over 4 years).
Works in both bull and bear markets by using volume confirmation to filter false breakouts and ATR stops to manage risk.
Weekly Donchian levels provide stronger structural support/resistance from higher timeframe.
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
    
    # Calculate weekly Donchian levels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle)
    
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
    breakout_level = 0.0  # Track breakout level for 50% retracement exit
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                breakout_level = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper AND volume spike
            if (price > upper and volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                breakout_level = upper  # Record breakout level for exit condition
            # Short: Price breaks below weekly Donchian lower AND volume spike
            elif (price < lower and volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                breakout_level = lower  # Record breakout level for exit condition
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces 50% of the breakout range
            if position == 1:
                retracement_level = breakout_level - 0.5 * (breakout_level - middle)
                if price <= retracement_level:
                    exit_signal = True
            elif position == -1:
                retracement_level = breakout_level + 0.5 * (middle - breakout_level)
                if price >= retracement_level:
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
                breakout_level = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyDonchian20_Breakout_VolumeConfirmation_ATRStop"
timeframe = "1d"
leverage = 1.0