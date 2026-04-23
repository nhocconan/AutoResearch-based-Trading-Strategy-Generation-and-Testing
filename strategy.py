#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme with 12h EMA50 Trend and Volume Spike Filter
- Williams %R identifies overbought/oversold conditions (-20/-80 levels)
- 12h EMA(50) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 2.0x 20-period average confirms strong reversal momentum
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years) to minimize fee drag
- Works in bull markets via buying oversold dips in uptrend, in bear markets via selling overbought rallies in downtrend
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
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Williams %R(14) on 12h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_12h['close'].values) / (highest_high - lowest_low)
    
    # Align Williams %R to 4h timeframe (completed 12h bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA12h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme signals with trend filter and volume spike
        # Long: Williams %R < -80 (oversold) + uptrend + volume spike
        # Short: Williams %R > -20 (overbought) + downtrend + volume spike
        long_signal = (williams_r_aligned[i] < -80 and 
                      close[i] > ema_50_12h_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r_aligned[i] > -20 and 
                       close[i] < ema_50_12h_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or Williams %R returns to neutral zone
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or Williams %R > -50 (leaving oversold)
                if (close[i] < ema_50_12h_aligned[i] or 
                    williams_r_aligned[i] > -50):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or Williams %R < -50 (leaving overbought)
                if (close[i] > ema_50_12h_aligned[i] or 
                    williams_r_aligned[i] < -50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0