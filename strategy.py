#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h Elder Ray (Bull/Bear Power) + 1d EMA200 trend filter + volume confirmation.
Long when: Bull Power > 0, price > 1d EMA200, and volume > 1.5x 20-period average.
Short when: Bear Power < 0, price < 1d EMA200, and volume > 1.5x 20-period average.
Elder Ray measures bull/bear strength relative to EMA13. Combined with 1d EMA200 trend filter,
this captures strong directional moves in both bull and bear markets while avoiding counter-trend trades.
Volume confirmation ensures breakouts have conviction.
Designed for low trade frequency (12-37/year) with high win rate in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Elder Ray (EMA13, Bull/Bear Power)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for EMA200 trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA13 on 12h
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema13_12h = ema(close_12h, 13)
    
    # Calculate Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power_12h = high_12h - ema13_12h
    bear_power_12h = low_12h - ema13_12h
    
    # Calculate 1d EMA200
    ema200_1d = ema(close_1d, 200)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = ema(volume_1d, 20)  # Using EMA for smoother average
    
    # Align all to primary timeframe (6h)
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_12h_aligned[i]) or 
            np.isnan(bear_power_12h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive, price above 1d EMA200, volume confirmed
            if (bull_power_12h_aligned[i] > 0 and 
                close[i] > ema200_1d_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, price below 1d EMA200, volume confirmed
            elif (bear_power_12h_aligned[i] < 0 and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power turns negative (momentum shift)
            if bear_power_12h_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power turns positive (momentum shift)
            if bull_power_12h_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hElderRay_1dEMA200_Volume_Confirm"
timeframe = "6h"
leverage = 1.0