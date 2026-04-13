#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v1
Hypothesis: Breakouts of Camarilla H3/L3 levels with volume > 1.5x 20-period average and
1d EMA200 trend filter. Captures institutional breakouts in trending markets while
avoiding false signals in ranging conditions. Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Previous period's high/low for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for current period using previous period's range
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        ema200_1d = np.full(len(prices), np.nan)
    else:
        close_1d = df_1d['close'].values
        ema200_1d_raw = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema200_1d = align_htf_to_ltf(prices, df_1d, ema200_1d_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema200_1d[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above Camarilla H3 with volume expansion and 1d uptrend
        long_signal = (close[i] > camarilla_h3[i] and 
                      volume_expansion[i] and 
                      close[i] > ema200_1d[i])
        
        # Short signal: break below Camarilla L3 with volume expansion and 1d downtrend
        short_signal = (close[i] < camarilla_l3[i] and 
                       volume_expansion[i] and 
                       close[i] < ema200_1d[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0