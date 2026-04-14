# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Williams %R mean reversion + 1d volume filter
# Williams %R identifies overbought/oversold conditions on higher timeframe
# When %R reaches extreme levels (>80 oversold, <20 overbought) with volume confirmation,
# mean reversion tends to occur. Works in both bull and bear markets as it captures
# exhaustion points rather than trends. Uses 1d Williams %R for signal and 1d volume
# for confirmation - avoids overtrading by requiring confluence.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14 periods)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    close_1d = df_1d['close'].values
    
    # Avoid division by zero
    dd = highest_high - lowest_low
    williams_r = np.where(dd != 0, (highest_high - close_1d) / dd * -100, -50)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume moving average (20 periods)
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need enough for Williams %R and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Mean reversion signals from Williams %R
        oversold = williams_r_aligned[i] <= -80  # Oversold condition
        overbought = williams_r_aligned[i] >= -20  # Overbought condition
        
        # Volume confirmation: current volume above average
        vol_confirmed = vol > vol_ma_aligned[i]
        
        if position == 0:
            # Enter long: oversold + volume confirmation
            if oversold and vol_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: overbought + volume confirmation
            elif overbought and vol_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral territory (-50) or overbought
            if williams_r_aligned[i] >= -50:  # Return to neutral or overbought
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral territory (-50) or oversold
            if williams_r_aligned[i] <= -50:  # Return to neutral or oversold
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dWilliamsR_Volume_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0