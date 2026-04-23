#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ATR ratio (ATR7/ATR30) < 0.8 (low vol regime) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND 1d ATR ratio < 0.8 AND volume > 1.5x 20-period average.
Exit when price returns to Donchian midpoint (mean reversion) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to balance return and risk. Targets 12-30 trades/year per symbol.
In low volatility regimes, breakouts are more likely to sustain; high volatility regimes often reverse quickly.
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
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d ATR regime filter (ATR7/ATR30 ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # first bar
    
    atr7_1d = pd.Series(tr_1d).rolling(window=7, min_periods=7).mean().values
    atr30_1d = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = atr7_1d / (atr30_1d + 1e-10)  # avoid division by zero
    
    # Align ATR ratio to 6h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation (using 6h data)
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
    start_idx = max(20, 30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        atr_ratio = atr_ratio_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper AND low volatility regime (ATR ratio < 0.8) AND volume spike
            if (price > upper and 
                atr_ratio < 0.8 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian lower AND low volatility regime AND volume spike
            elif (price < lower and 
                  atr_ratio < 0.8 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price returns to Donchian midpoint (mean reversion)
            if position == 1 and price <= mid:
                exit_signal = True
            elif position == -1 and price >= mid:
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

name = "6H_Donchian20_1dATRRegime_VolumeSpike_ATRStop"
timeframe = "6h"
leverage = 1.0