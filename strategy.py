#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R Extreme with 1w EMA50 Trend and Volume Confirmation
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought
- 1w EMA(50) provides higher timeframe trend filter for alignment
- Volume confirmation (> 2.0x 20-period average) ensures breakout validity
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years) to minimize fee drag
- Mean reversion from extremes in direction of weekly trend works in both bull and bear markets
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We need 1d high/low/close for the calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R with proper min_periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 1d timeframe (no additional delay needed as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA1w, volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme signals with trend filter and volume confirmation
        # Long: Williams %R < -80 (oversold) + uptrend + volume confirmation
        # Short: Williams %R > -20 (overbought) + downtrend + volume confirmation
        long_signal = (williams_r_aligned[i] < -80 and 
                      close[i] > ema_50_1w_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r_aligned[i] > -20 and 
                       close[i] < ema_50_1w_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral territory or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R > -50 (return to neutral) or trend reversal
                if (williams_r_aligned[i] > -50 or 
                    close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R < -50 (return to neutral) or trend reversal
                if (williams_r_aligned[i] < -50 or 
                    close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_Extreme_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0