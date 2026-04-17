#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Williams %R extreme filter + volume confirmation.
Long when weekly Williams %R < -80 (oversold) and daily close > daily open (bullish candle) and volume > 1.5x 20-day average.
Short when weekly Williams %R > -20 (overbought) and daily close < daily open (bearish candle) and volume > 1.5x 20-day average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 30-100 total trades over 4 years.
Williams %R identifies overextended moves on weekly chart; volume confirms participation; daily candle direction ensures alignment with short-term momentum.
Designed to work in bull markets (buy oversold dips) and bear markets (sell overbought rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and candle direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for Williams %R
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close_1w) / (highest_high - lowest_low)) * -100,
                          -50)  # neutral when no range
    
    # Align all to 1d
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # need enough for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        # Daily candle direction
        bullish_candle = close[i] > open_prices[i]
        bearish_candle = close[i] < open_prices[i]
        
        if position == 0:
            # Long: weekly oversold + bullish daily candle + volume
            if (williams_r_aligned[i] < -80 and 
                bullish_candle and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: weekly overbought + bearish daily candle + volume
            elif (williams_r_aligned[i] > -20 and 
                  bearish_candle and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly Williams %R returns above -50 (neutral) or bearish engulfing
            if williams_r_aligned[i] > -50 or bearish_candle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly Williams %R returns below -50 (neutral) or bullish engulfing
            if williams_r_aligned[i] < -50 or bullish_candle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wWilliamsR_Volume_CandleConfirm"
timeframe = "1d"
leverage = 1.0