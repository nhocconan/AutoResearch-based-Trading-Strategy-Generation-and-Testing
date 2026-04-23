#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend Filter + Volume Spike
- Williams %R identifies overbought/oversold extremes; reversals from these levels offer high-probability entries
- 1d EMA34 defines the intermediate trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 2.0x 20-period MA) reduces false reversals
- Designed for 6h timeframe to capture medium-term reversals with controlled frequency
- Works in bull via oversold bounces and in bear via overbought rejections
- Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average (approx 5 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA34 AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA34 AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) OR crosses 1d EMA34
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R > -50 OR price < 1d EMA34
                if williams_r_aligned[i] > -50 or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R < -50 OR price > 1d EMA34
                if williams_r_aligned[i] < -50 or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0