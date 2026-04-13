#!/usr/bin/env python3
"""
4h_1d_Keltner_Channel_Breakout
Hypothesis: Trade breakouts from daily Keltner Channel on 4h timeframe with volume confirmation.
Keltner Channel adapts to volatility, providing dynamic support/resistance. 
Breakouts above upper channel signal bullish momentum; breakdowns below lower channel signal bearish momentum.
Volume confirmation filters out low-participation moves. Designed to work in both bull and bear markets
by capturing momentum bursts during regime shifts.
Target: 25-35 trades/year to minimize fee drag.
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
    
    # Get daily data for Keltner Channel (ATR-based bands)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) for daily data
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_10_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean()
    
    # Calculate EMA(20) of close for daily data
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Calculate Keltner Channel: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    kc_upper_1d = ema_20_1d + (2 * atr_10_1d)
    kc_lower_1d = ema_20_1d - (2 * atr_10_1d)
    
    # Align Keltner Channel to 4h
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper_1d.values)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower_1d.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average (moderate filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above upper Keltner Channel with volume expansion
        long_condition = (close[i] > kc_upper_aligned[i]) and volume_expansion[i]
        
        # Short: breakdown below lower Keltner Channel with volume expansion
        short_condition = (close[i] < kc_lower_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Keltner_Channel_Breakout"
timeframe = "4h"
leverage = 1.0