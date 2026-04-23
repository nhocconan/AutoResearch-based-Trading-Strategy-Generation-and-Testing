#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme with 1d EMA34 Trend Filter and Volume Spike
- Williams %R(14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
- Only take extreme readings: Williams %R < -90 for long, > -10 for short to avoid churn
- 1d EMA34 defines medium-term trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 2.0x 20-period average) ensures breakout strength
- Designed for 4h timeframe targeting 20-30 trades/year (80-120 over 4 years) to avoid overtrading
- Extreme Williams %R readings + trend filter + volume spike = high-probability entries
- Works in both bull and bear markets by following the 1d EMA34 trend filter
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
    
    # Calculate 1d Williams %R(14) - needs 14 periods
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # need 14 for Williams %R + 1 for current
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(15, 34, 20)  # need Williams %R(14), 1d EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -90 (extreme oversold) AND price > 1d EMA34 AND volume spike
            if (williams_r_aligned[i] < -90 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (extreme overbought) AND price < 1d EMA34 AND volume spike
            elif (williams_r_aligned[i] > -10 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 1d EMA34 OR Williams %R returns to neutral zone (-50 to -50)
            exit_signal = False
            if position == 1:
                # Exit long when price < 1d EMA34 OR Williams %R > -50 (no longer oversold)
                if close[i] < ema_34_1d_aligned[i] or williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > 1d EMA34 OR Williams %R < -50 (no longer overbought)
                if close[i] > ema_34_1d_aligned[i] or williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0