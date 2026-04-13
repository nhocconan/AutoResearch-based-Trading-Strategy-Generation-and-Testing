#!/usr/bin/env python3
"""
12h_1D_Camarilla_Breakout_Volume_Confirmation_v2
Hypothesis: Buy when price breaks above daily Camarilla H4 level with volume > 2.5x 50-period average and close above open (bullish candle), sell when price breaks below daily L4 level with volume > 2.5x 50-period average and close below open (bearish candle). Uses 12h primary timeframe with 1d trend filter. Designed to work in both bull and bear markets by capturing genuine breakouts with strong volume and directional confirmation. Volume threshold increased to 2.5x to reduce trade frequency and improve signal quality. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 2.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 2.5)
    
    # Previous day's high/low/close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    camarilla_h4_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align daily levels to 12h timeframe (wait for daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above daily Camarilla H4 with volume expansion and bullish candle
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > open_price[i])
        
        # Short signal: break below daily Camarilla L4 with volume expansion and bearish candle
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < open_price[i])
        
        if position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1 and short_signal:
            position = -1
            signals[i] = -position_size
        elif position == -1 and long_signal:
            position = 1
            signals[i] = position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1D_Camarilla_Breakout_Volume_Confirmation_v2"
timeframe = "12h"
leverage = 1.0