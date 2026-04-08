#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (previous week's values)
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2 * pivot_weekly - low_weekly
    s1_weekly = 2 * pivot_weekly - high_weekly
    
    # Align pivot levels to 1d timeframe
    pivot_daily = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r1_daily = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_daily = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Daily trend: 34-period EMA
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34[i]) or np.isnan(pivot_daily[i]) or np.isnan(r1_daily[i]) or 
            np.isnan(s1_daily[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S1 or trend fails
            if close[i] < s1_daily[i] or close[i] < ema_34[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R1 or trend fails
            if close[i] > r1_daily[i] or close[i] > ema_34[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_34[i]
            bearish = close[i] < ema_34[i]
            
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