#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Williams %R extremes with volume confirmation and ATR trailing stop.
Long when 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-period average.
Short when 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-period average.
Exit when price retraces to 1d Williams %R midpoint (-50) or ATR trailing stop hit (2.5*ATR from highest/lowest since entry).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 12h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams %R provides mean-reversion signals in ranging markets while volume confirmation filters false signals.
ATR trailing stop manages risk during trending periods. Works in both bull and bear markets by adapting to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R midpoint (-50) for mean reversion exit
    williams_mid = -50 * np.ones_like(williams_r)
    
    # Align Williams %R levels to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    williams_mid_aligned = align_htf_to_ltf(prices, df_1d, williams_mid)
    
    # Volume average (20-period) on 12h timeframe
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
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(williams_mid_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr_val = williams_r_aligned[i]
        mid_val = williams_mid_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND volume spike
            if (wr_val < -80 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Williams %R overbought (> -20) AND volume spike
            elif (wr_val > -20 and volume[i] > 1.5 * vol_ma_val):
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
            
            # Primary exit: Price retraces to Williams %R midpoint (-50) equivalent price level
            # We approximate this as a mean reversion to the recent average price
            # Using the midpoint as a dynamic level based on recent price action
            if position == 1 and price <= mid_val:
                exit_signal = True
            elif position == -1 and price >= mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
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

name = "12H_WilliamsR_Extremes_VolumeConfirmation_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0