#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day volatility filter and volume confirmation.
Long when price breaks above Donchian upper band with low volatility and volume spike.
Short when price breaks below Donchian lower band with low volatility and volume spike.
Exit when price crosses midline (median of high-low-close).
Designed for low trade frequency (<40/year) by requiring volatility contraction
before breakout, which reduces false signals. Works in both bull and bear markets
by capturing volatility breakouts that often precede strong moves.
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
    
    # Load 1-day data for volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # 1-day ATR for volatility filter (ATR < 20-period average = low volatility)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first value
    tr3[0] = tr1[0]  # first value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: current ATR < 20-period ATR average (low volatility environment)
        low_vol = atr_1d_aligned[i] < atr_ma_20_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with low volatility and volume spike
            if (close[i] > high_max_20[i] and low_vol and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with low volatility and volume spike
            elif (close[i] < low_min_20[i] and low_vol and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses midline (mean reversion within the channel)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midline
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midline
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_VolatilityFilter_Volume"
timeframe = "4h"
leverage = 1.0