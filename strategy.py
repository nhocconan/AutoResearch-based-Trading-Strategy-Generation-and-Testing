#!/usr/bin/env python3
"""
1d_1w_AntiBubble_Breakout
Hypothesis: In crypto, multi-month bubbles often form then burst. We detect when price exceeds 
a 20-week exponential decay envelope (resembling bubble top) and go short, or falls below 
the same envelope (bubble bottom) and go long. Uses 1d timeframe with 1w HTF for envelope.
Works in both bull (buying dips) and bear (selling rallies) by fading extreme deviations 
from the long-term trend. Includes volume confirmation to avoid false signals.
Target: 10-25 trades/year on 1d (40-100 total over 4 years).
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
    
    # Get weekly data for bubble envelope
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 20-week EMA as bubble center
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-week standard deviation for bubble width
    # Using rolling window on weekly data
    close_1w_series = pd.Series(close_1w)
    std_20 = close_1w_series.rolling(window=20, min_periods=20).std().values
    
    # Bubble envelope: EMA ± 2.0 * std (adjustable)
    upper_env = ema_20 + (2.0 * std_20)
    lower_env = ema_20 - (2.0 * std_20)
    
    # Align envelope to daily timeframe (wait for weekly close)
    upper_env_aligned = align_htf_to_ltf(prices, df_1w, upper_env)
    lower_env_aligned = align_htf_to_ltf(prices, df_1w, lower_env)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: weekly volume > 1.5x 20-week average
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume_1w > (vol_ma_20 * 1.5)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1w, volume_expansion)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_env_aligned[i]) or 
            np.isnan(lower_env_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price below lower envelope with volume expansion
        if close[i] < lower_env_aligned[i] and volume_expansion_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short signal: price above upper envelope with volume expansion
        elif close[i] > upper_env_aligned[i] and volume_expansion_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Exit when price returns to center (EMA) - take profit on mean reversion
        elif position == 1 and close[i] > ema_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < ema_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        # Hold position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_AntiBubble_Breakout"
timeframe = "1d"
leverage = 1.0