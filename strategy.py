#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R extreme reversal with 1-day EMA50 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) with price above 1-day EMA50 and volume spike.
Short when Williams %R > -20 (overbought) with price below 1-day EMA50 and volume spike.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Williams %R identifies overextended moves; 1-day EMA50 filters for trend alignment;
volume spike confirms conviction. Designed for low trade frequency by requiring
extreme readings and multiple confirmations. Works in both bull and bear markets
by fading extremes in the direction of the higher timeframe trend.
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
    
    # Lookback period for Williams %R
    lookback = 14
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with price above EMA50 and volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema50_1d_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with price below EMA50 and volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema50_1d_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if williams_r[i] > -50 and williams_r[i-1] <= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50
                if williams_r[i] < -50 and williams_r[i-1] >= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Extreme_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0