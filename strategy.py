#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 4h ATR-based volatility filter and volume confirmation.
# Uses price channel breakouts with volatility filtering to avoid choppy markets.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Target: 20-40 trades/year to avoid fee drag.

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
    
    # Get 4h data for ATR calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Donchian channel (20-period) on price data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # Volatility filter: ATR > 0.5 * 50-period ATR average (avoid low volatility chop)
    atr_50 = pd.Series(atr_4h_aligned).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_4h_aligned > (atr_50 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_4h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: break above 20-period high with volume and volatility
        long_breakout = close[i] > high_20[i]
        long_entry = long_breakout and volume_filter[i] and vol_filter[i]
        
        # Short: break below 20-period low with volume and volatility
        short_breakout = close[i] < low_20[i]
        short_entry = short_breakout and volume_filter[i] and vol_filter[i]
        
        # Exit conditions: opposite Donchian level (mean reversion)
        long_exit = close[i] < low_20[i] and position == 1
        short_exit = close[i] > high_20[i] and position == -1
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_VolVol_Filter"
timeframe = "4h"
leverage = 1.0