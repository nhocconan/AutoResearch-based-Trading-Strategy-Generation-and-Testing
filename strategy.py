#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d ATR-based volatility expansion filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day high with ATR(7)/ATR(30) > 1.5 (volatility expansion) and volume > 1.5x 20-period volume average.
Short when price breaks below 20-day low with ATR(7)/ATR(30) > 1.5 and volume > 1.5x 20-period volume average.
Volatility expansion helps capture strong momentum moves after consolidation, working in both bull and bear markets.
ATR stoploss exits when price retraces 2*ATR from extreme.
Designed for fewer, higher-quality trades to avoid fee drag.
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
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(7) and ATR(30)
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period TR is just high-low
        atr_vals = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr_vals
    
    atr_7_1d = atr(high_1d, low_1d, close_1d, 7)
    atr_30_1d = atr(high_1d, low_1d, close_1d, 30)
    
    # Calculate 1d Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_1d, low_1d, 20)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    atr_7_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_7_1d)
    atr_30_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_30_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    atr_multiplier = 2.0  # ATR multiplier for stoploss
    long_extreme = 0.0    # track highest high since entering long
    short_extreme = 0.0   # track lowest low since entering short
    
    start_idx = 40  # need enough for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_7_1d_aligned[i]) or 
            np.isnan(atr_30_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion: ATR(7) > 1.5 * ATR(30)
        vol_expansion = atr_7_1d_aligned[i] > 1.5 * atr_30_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with vol expansion and volume
            if (close[i] > donchian_upper_aligned[i] and 
                vol_expansion and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]  # initialize extreme to current high
            # Short: price breaks below 20-day low with vol expansion and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  vol_expansion and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]  # initialize extreme to current low
        
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            # ATR stoploss: exit if price retraces 2*ATR from extreme
            if close[i] < long_extreme - atr_multiplier * atr_7_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            # ATR stoploss: exit if price rallies 2*ATR from extreme
            if close[i] > short_extreme + atr_multiplier * atr_7_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dATRratio_VolatilityExpansion_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0