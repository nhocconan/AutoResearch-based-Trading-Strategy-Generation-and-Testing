#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly high/low for channel
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align to 12h timeframe
    weekly_high_12h = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_12h = align_htf_to_ltf(prices, df_1w, weekly_low)
    daily_pivot_12h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_12h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_12h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Weekly ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need weekly ATR and other indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_12h[i]) or 
            np.isnan(weekly_low_12h[i]) or 
            np.isnan(daily_pivot_12h[i]) or 
            np.isnan(daily_r1_12h[i]) or 
            np.isnan(daily_s1_12h[i]) or 
            np.isnan(atr_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Price relative to weekly channel
        price_near_high = close[i] >= (weekly_high_12h[i] - 0.5 * atr_12h[i])
        price_near_low = close[i] <= (weekly_low_12h[i] + 0.5 * atr_12h[i])
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_12h[i]
        price_below_s1 = close[i] < daily_s1_12h[i]
        
        if position == 0:
            # Long: Price near weekly high AND breaks above daily R1 with volume
            if (price_near_high and price_above_r1 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price near weekly low AND breaks below daily S1 with volume
            elif (price_near_low and price_below_s1 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR back into weekly channel
            if (close[i] < daily_pivot_12h[i]) or (close[i] < weekly_low_12h[i] + atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR back into weekly channel
            if (close[i] > daily_pivot_12h[i]) or (close[i] > weekly_high_12h[i] - atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyChannel_DailyPivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0