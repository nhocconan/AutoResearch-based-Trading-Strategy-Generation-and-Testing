#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R extreme reversal with 1w EMA34 trend filter and volume confirmation
- Uses Williams %R(14) from 1d for extreme oversold/overbought signals
- 1w EMA(34) defines major trend direction (only trade with trend)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum reversals
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years)
- Works in both bull and bear markets by trading mean reversions in trending markets
- Williams %R provides clear extreme levels for entry
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
    
    # Williams %R calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        0
    )
    
    # Align Williams %R to 1d timeframe (already aligned since we're using 1d data)
    williams_r_aligned = williams_r  # No alignment needed for same timeframe
    
    # Calculate 1w EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Williams %R extreme conditions
        williams_oversold = williams_r_aligned[i] < -80  # Extreme oversold
        williams_overbought = williams_r_aligned[i] > -20  # Extreme overbought
        
        # Trend filter: price above/below 1w EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long conditions: Williams %R oversold, price above 1w EMA (uptrend), volume spike
            long_signal = (williams_oversold and 
                          price_above_ema and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: Williams %R overbought, price below 1w EMA (downtrend), volume spike
            short_signal = (williams_overbought and 
                           price_below_ema and
                           volume[i] > 1.5 * vol_ma[i])
            
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
                # Exit long: Williams %R rises above -50 or price falls below 1w EMA
                if (williams_r_aligned[i] > -50 or 
                    price_below_ema):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R falls below -50 or price rises above 1w EMA
                if (williams_r_aligned[i] < -50 or 
                    price_above_ema):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_Extreme_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0