#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d ATR volatility regime filter + 4h Donchian(20) breakout + volume confirmation.
Long when price breaks above 4h Donchian(20) upper band with 1d ATR ratio (ATR7/ATR30) > 1.2 (expanding volatility) and volume > 1.5x 20-period 4h volume average.
Short when price breaks below 4h Donchian(20) lower band with same volatility and volume conditions.
Volatility expansion captures breakout momentum in both bull and bear markets; Donchian provides structural breakout levels; volume confirms participation.
Designed to avoid choppy markets (low ATR ratio) and false breakouts. Target: 50-150 total trades over 4 years.
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
    
    # Get 4h data for Donchian breakout and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for ATR volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_arr, low_arr, window):
        upper = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper_4h, donchian_lower_4h = donchian_channel(high_4h, low_4h, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(7) and ATR(30) for volatility regime
    def atr(high_arr, low_arr, close_arr, window):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr2[0] = tr1[0]  # first bar: no previous close
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_val = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_val
    
    atr_7_1d = atr(high_1d, low_1d, close_1d, 7)
    atr_30_1d = atr(high_1d, low_1d, close_1d, 30)
    atr_ratio_1d = atr_7_1d / atr_30_1d  # >1.2 indicates expanding volatility
    
    # Align all to 6h
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # need enough for Donchian and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: expanding volatility (ATR7/ATR30 > 1.2)
        volatility_expanding = atr_ratio_1d_aligned[i] > 1.2
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume_4h_aligned[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with expanding volatility and volume
            if (close[i] > donchian_upper_4h_aligned[i] and 
                volatility_expanding and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian lower with expanding volatility and volume
            elif (close[i] < donchian_lower_4h_aligned[i] and 
                  volatility_expanding and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian lower (reversal signal)
            if close[i] < donchian_lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian upper (reversal signal)
            if close[i] > donchian_upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dATR_VolRegime_4hDonchian20_Volume_Confirm"
timeframe = "6h"
leverage = 1.0