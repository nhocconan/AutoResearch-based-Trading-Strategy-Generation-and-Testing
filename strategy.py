#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme + 1d EMA34 Trend Filter + Volume Spike
- Williams %R(14): Long when < -80 (oversold) and price > 1d EMA34 (uptrend filter)
                    Short when > -20 (overbought) and price < 1d EMA34 (downtrend filter)
- Volume confirmation: > 2.0x 20-period average to avoid false reversals
- Uses 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in bull markets via buying dips in uptrend, in bear markets via selling rallies in downtrend
- Williams %R provides mean-reversion edge while EMA34 filters for trend alignment
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams %R(14) on 4h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, volume MA 20, Williams %R 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        close_1d = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema34_aligned[i]
        downtrend = close_1d_aligned[i] < ema34_aligned[i]
        
        # Williams %R signals with trend filter and volume confirmation
        # Long: Oversold (%R < -80) + uptrend + volume spike
        # Short: Overbought (%R > -20) + downtrend + volume spike
        long_signal = (williams_r[i] < -80 and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r[i] > -20 and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R reverts to midpoint or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: %R rises above -50 (momentum fading) or trend turns down
                if (williams_r[i] > -50 or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: %R falls below -50 (momentum fading) or trend turns up
                if (williams_r[i] < -50 or 
                    not downtrend):
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