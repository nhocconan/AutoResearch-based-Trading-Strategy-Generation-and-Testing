#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian breakout with 1-day ATR volatility filter and volume confirmation.
Trades breakouts of 20-period Donchian channels when volatility is low (reducing false breakouts)
and volume confirms institutional interest. Designed for low trade frequency (15-30 trades/year)
to minimize fee flood and work in both bull and bear markets by using ATR-based volatility
regime filtering to avoid choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for ATR filter and Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # First period
    tr3[0] = tr1[0]  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume spike: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is below its 50-period median (low vol regime)
        if i >= 50:
            atr_median_50 = np.nanmedian(atr_14_1d_aligned[i-50:i+1])
            low_vol_regime = atr_14_1d_aligned[i] < atr_median_50
        else:
            low_vol_regime = True  # Not enough data for median, allow trading
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0 and low_vol_regime and vol_spike:
            # Long: price breaks above Donchian high
            if close[i] > donch_high_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low
            elif close[i] < donch_low_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level or volatility increases
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low or high volatility
                if close[i] < donch_low_20_aligned[i] or not low_vol_regime:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high or high volatility
                if close[i] > donch_high_20_aligned[i] or not low_vol_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR14_Volume"
timeframe = "12h"
leverage = 1.0