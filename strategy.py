#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme + 12h EMA50 trend filter + volume spike
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought
- 12h EMA50 defines higher timeframe trend: only trade reversals in trend direction (long in uptrend, short in downtrend)
- Volume confirmation (> 1.8x 20-period average) filters false reversals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 12h trend
- Williams %R extremes provide mean reversion signals that work well in ranging markets
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
    
    # Calculate Williams %R(14) on primary timeframe
    if len(close) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # for EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) with 12h uptrend and volume spike
            long_signal = (williams_r[i] < -80 and 
                          close[i] > ema_50_12h_aligned[i] and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: Williams %R overbought (> -20) with 12h downtrend and volume spike
            short_signal = (williams_r[i] > -20 and 
                           close[i] < ema_50_12h_aligned[i] and
                           volume[i] > 1.8 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Williams %R extreme or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R overbought (> -20) or 12h trend turns bearish
                if (williams_r[i] > -20 or 
                    close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R oversold (< -80) or 12h trend turns bullish
                if (williams_r[i] < -80 or 
                    close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0