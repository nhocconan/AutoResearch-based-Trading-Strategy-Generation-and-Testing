#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_breakout_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (previous week's values)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align pivot levels to daily timeframe
    pivot_daily = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_daily = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_daily = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 1w trend: 21-period EMA
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_daily = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_daily[i]) or np.isnan(pivot_daily[i]) or 
            np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S1 or trend fails
            if close[i] < s1_daily[i] or close[i] < ema_21_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R1 or trend fails
            if close[i] > r1_daily[i] or close[i] > ema_21_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_21_daily[i]
            bearish = close[i] < ema_21_daily[i]
            
            # Long: price > R1 + bullish trend + volume
            if (close[i] > r1_daily[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < S1 + bearish trend + volume
            elif (close[i] < s1_daily[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals