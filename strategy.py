#!/usr/bin/env python3
"""
1h_4h_1D_Range_Breakout_With_Volume_Confirmation
Hypothesis: In 1h timeframe, buy when price breaks above 4h Donchian high (20) with volume > 1.5x 20-period average and close > 4h EMA20, sell when price breaks below 4h Donchian low (20) with volume confirmation and close < 4h EMA20. Uses 4h for trend/structure and 1h for precise entry. Volume filter reduces false breakouts. Designed for 15-30 trades/year to avoid fee drag. Works in bull/bear by trading breakouts with institutional volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # 4h data for structure and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA20 trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: break above 4h Donchian high with volume expansion and close > 4h EMA20
        long_signal = (close[i] > donchian_high_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema20_4h_aligned[i])
        
        # Short: break below 4h Donchian low with volume expansion and close < 4h EMA20
        short_signal = (close[i] < donchian_low_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema20_4h_aligned[i])
        
        # Exit on opposite signal
        if position == 1 and short_signal:
            position = -1
            signals[i] = -position_size
        elif position == -1 and long_signal:
            position = 1
            signals[i] = position_size
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h_1D_Range_Breakout_With_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0