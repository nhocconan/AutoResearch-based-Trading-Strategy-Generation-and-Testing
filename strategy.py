#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_Regime
Hypothesis: Combines Camarilla pivot levels from 1d with volume confirmation and a Chop Zone regime filter.
In trending markets (Chop < 38.2), takes breakout trades at Camarilla H4/L4 levels with volume > 1.5x average.
In ranging markets (Chop > 61.8), fades touches at H3/L3 levels. Uses 4h timeframe for signal generation.
Designed to work in both bull and bear markets by adapting to volatility regime.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
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
    
    # Get daily data for Camarilla and Chop Zone
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    rng = high_1d - low_1d
    H4 = close_1d + 1.1 * rng / 2
    L4 = close_1d - 1.1 * rng / 2
    H3 = close_1d + 1.1 * rng / 4
    L3 = close_1d - 1.1 * rng / 4
    
    # Chop Zone: EMA of |log(high/low)|, then double smoothed
    # Highly volatile = trending (low Chop), low volatility = ranging (high Chop)
    hl_ratio = np.log(high_1d / low_1d)
    abs_hl_ratio = np.abs(hl_ratio)
    ema1 = pd.Series(abs_hl_ratio).ewm(span=14, adjust=False, min_periods=14).mean()
    chop = 100 * pd.Series(ema1).ewm(span=14, adjust=False, min_periods=14).mean() / \
           pd.Series(abs_hl_ratio).ewm(span=14, adjust=False, min_periods=14).mean()
    chop = chop.values
    
    # Get 4h data for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_expansion_4h = volume_4h > (vol_ma_20_4h * 1.5)
    
    # Align all signals to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_4h, volume_expansion_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        vol_exp = volume_expansion_aligned[i]
        
        # Regime-based logic
        if chop_val < 38.2:  # Trending regime - breakout at H4/L4
            # Long breakout above H4 with volume
            if close_4h[i] > H4_aligned[i] and vol_exp:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short breakdown below L4 with volume
            elif close_4h[i] < L4_aligned[i] and vol_exp:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold existing position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
                
        elif chop_val > 61.8:  # Ranging regime - fade at H3/L3
            # Short at H3 resistance
            if close_4h[i] > H3_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Long at L3 support
            elif close_4h[i] < L3_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Hold existing position
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:  # Transition zone - stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0