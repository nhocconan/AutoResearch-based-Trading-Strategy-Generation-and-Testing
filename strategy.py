#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversion with 1d Trend Filter and Volume Spike
- Williams %R measures overbought/oversold levels; extreme readings (>80 or < -20) often precede reversals
- 1d EMA(50) ensures alignment with higher timeframe trend to avoid fighting the daily trend
- Volume spike (>2.0x 20-period average) confirms strong participation at turning points
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading mean reversion extremes within the dominant daily trend
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA1d, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme reversion signals with trend filter
        # Long: oversold (< -80) + uptrend + volume spike
        # Short: overbought (> -20) + downtrend + volume spike
        long_signal = (williams_r[i] < -80 and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r[i] > -20 and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 (leaving oversold) or trend reversal
                if (williams_r[i] > -50 or 
                    close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R falls below -50 (leaving overbought) or trend reversal
                if (williams_r[i] < -50 or 
                    close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0