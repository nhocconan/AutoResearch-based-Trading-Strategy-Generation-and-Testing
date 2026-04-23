#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above Donchian upper band AND ATR(1d) > SMA(ATR(1d), 50) AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND ATR(1d) > SMA(ATR(1d), 50) AND volume > 2.0x 20-period average.
Exit when price retraces to Donchian midpoint or ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profit potential.
Donchian breakouts capture strong momentum moves, and the 1d ATR regime filter ensures we only trade
when volatility is expanding (avoiding choppy markets). Volume confirmation ensures institutional participation.
Works in bull markets (breakouts with volume in expanding volatility) and bear markets (breakdowns with volume
in expanding volatility) by following the breakout direction with volatility filter.
Target trade frequency: 12-30 trades/year per symbol (50-120 total over 4 years) to avoid fee drag.
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
    
    # Calculate 1d ATR(14) and its 50-period SMA for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 65:  # Need 14 for ATR + 50 for SMA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50-period SMA of ATR for regime filter (expanding volatility)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian(20) bands on 12h timeframe
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    midpoint = (upper_band + lower_band) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation on 12h
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
    start_idx = max(65, donchian_window, 20)  # ATR regime needs 65, Donchian 20, vol MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(midpoint[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_ma_1d_val = atr_ma_1d_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        mid = midpoint[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND expanding volatility (ATR > MA) AND volume spike
            if close[i] > upper and atr_1d_val > atr_ma_1d_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower AND expanding volatility (ATR > MA) AND volume spike
            elif close[i] < lower and atr_1d_val > atr_ma_1d_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
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
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and close[i] <= mid:
                exit_signal = True
            elif position == -1 and close[i] >= mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dATR_Regime_VolumeConfirmation_MidExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0