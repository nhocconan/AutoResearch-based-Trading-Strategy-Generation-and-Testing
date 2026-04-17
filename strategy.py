#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w ATR volatility regime filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day high with 1w ATR(14)/ATR(50) < 0.8 (low volatility regime) and volume > 1.5x 20-day volume average.
Short when price breaks below 20-day low with 1w ATR(14)/ATR(50) < 0.8 and volume > 1.5x 20-day volume average.
Low volatility breakouts often lead to sustained moves as volatility expands after contraction.
Designed to work in both bull and bear markets by capturing volatility expansion breakouts.
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
    
    # Get 1w data for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w ATR(14) and ATR(50)
    def atr(high_vals, low_vals, close_vals, window):
        tr1 = high_vals - low_vals
        tr2 = np.abs(high_vals - np.roll(close_vals, 1))
        tr3 = np.abs(low_vals - np.roll(close_vals, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period TR is just high-low
        atr_vals = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr_vals
    
    atr_14_1w = atr(high_1w, low_1w, close_1w, 14)
    atr_50_1w = atr(high_1w, low_1w, close_1w, 50)
    
    # Calculate ATR ratio (short-term / long-term volatility)
    atr_ratio_1w = np.where(atr_50_1w > 0, atr_14_1w / atr_50_1w, 1.0)
    
    # Calculate 1d Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_1d, low_1d, 20)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (1d)
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_ratio_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR ratio < 0.8 (contraction)
        low_vol_regime = atr_ratio_1w_aligned[i] < 0.8
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with low volatility and volume
            if (close[i] > donchian_upper_aligned[i] and 
                low_vol_regime and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with low volatility and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  low_vol_regime and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wATRratio_Donchian20_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0