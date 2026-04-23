#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold (long), > -20 = overbought (short)
- 1d EMA(50) defines major trend: only long when price > EMA, short when price < EMA
- Volume confirmation (> 1.8x 20-period average) ensures breakout momentum
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading reversals within the 1d trend
- Williams %R is effective in ranging markets and catches reversals in trending markets
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
    
    # Calculate daily Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # Need min_periods for Williams %R
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = -(Highest High - Close) / (Highest High - Lowest Low) * 100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -((highest_high - close_1d) / (highest_high - lowest_low)) * 100,
        -50  # Neutral when range is zero
    )
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(15, 50, 20)  # Williams %R, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: Williams %R oversold, uptrend, volume spike
            long_signal = (oversold and 
                          uptrend and
                          volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: Williams %R overbought, downtrend, volume spike
            short_signal = (overbought and 
                           downtrend and
                           volume[i] > 1.8 * vol_ma[i])
            
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
                # Exit long: Williams %R rises above -50 or trend turns down
                if (williams_r_aligned[i] > -50 or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R falls below -50 or trend turns up
                if (williams_r_aligned[i] < -50 or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0