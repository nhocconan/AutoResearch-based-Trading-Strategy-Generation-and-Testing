#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 1-week EMA trend and volume confirmation.
Long when price breaks above Donchian high with 1-week EMA rising and volume spike.
Short when price breaks below Donchian low with 1-week EMA falling and volume spike.
Exit when price crosses the 10-period EMA in the opposite direction.
This strategy captures trend continuation with proper filtration to reduce whipsaw and limit trades.
Designed for low trade frequency (<40/year) by requiring multiple confirmations.
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
    
    # Load 1-week data for EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA10 for exit signal
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema10[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with 1-week EMA rising and volume spike
            if (close[i] > high_roll[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with 1-week EMA falling and volume spike
            elif (close[i] < low_roll[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 10-period EMA in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below EMA10
                if close[i] < ema10[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above EMA10
                if close[i] > ema10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_20_1wEMA20_Trend_Volume"
timeframe = "4h"
leverage = 1.0