#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with volume confirmation and ATR trailing stop.
Long when price breaks above Donchian(20) high AND volume > 1.3x 20-period average.
Short when price breaks below Donchian(20) low AND volume > 1.3x 20-period average.
Exit when price retraces 50% of the breakout move or ATR trailing stop (2.5*ATR) hit.
Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe
to target 20-50 trades/year per symbol (80-200 total over 4 years).
Volume confirmation filters false breakouts in ranging markets. ATR trailing stop
adjusts to volatility and locks in profits during trends. Works in both bull and
bear markets by capturing breakouts with institutional volume and managing risk.
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
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    highest = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = highest
    donchian_low = lowest
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume average (20-period) on 4h timeframe
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band AND volume confirmation
            if (price > upper and volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below Donchian lower band AND volume confirmation
            elif (price < lower and volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop logic
            if position == 1:
                if price > highest_since_entry:
                    highest_since_entry = price
            else:  # position == -1
                if price < lowest_since_entry:
                    lowest_since_entry = price
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces 50% of breakout move to mid-band
            if position == 1 and price <= mid:
                exit_signal = True
            elif position == -1 and price >= mid:
                exit_signal = True
            
            # ATR trailing stop: 2.5 * ATR from extreme point
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
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

name = "4H_Donchian20_VolumeConfirmation_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0