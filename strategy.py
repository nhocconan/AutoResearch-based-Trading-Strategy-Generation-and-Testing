#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 12h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND close < 12h EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to Donchian midpoint or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profit potential.
Donchian channels provide objective breakout levels with proven effectiveness in both bull and bear markets.
12h EMA50 filter ensures alignment with medium-term trend, reducing whipsaws in choppy markets.
Volume spike confirms institutional participation. Target trade frequency: 15-35 trades/year per symbol (60-140 total over 4 years) to avoid fee drag.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) from 4h OHLC
    if len(prices) < 20:
        return np.zeros(n)
    
    # Donchian upper band = 20-period high
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower band = 20-period low
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint = (upper + lower) / 2
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
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
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_12h_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        midpoint = donchian_mid[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND uptrend (price > EMA50) AND volume spike
            if close[i] > upper_band and close[i] > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower band AND downtrend (price < EMA50) AND volume spike
            elif close[i] < lower_band and close[i] < ema50_val and volume[i] > 1.8 * vol_ma_val:
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
            if position == 1 and close[i] <= midpoint:
                exit_signal = True
            elif position == -1 and close[i] >= midpoint:
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirmation_MidpointExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0