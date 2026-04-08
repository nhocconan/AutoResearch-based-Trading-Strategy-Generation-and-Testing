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
    
    # 1w data for trend filter (EMA21)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 1d data for pivot point calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using weekly high/low/close)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Pivot levels as arrays
    pivot_val = pivot
    r1_val = r1
    s1_val = s1
    r2_val = r2
    s2_val = s2
    
    # Align pivot levels to daily timeframe (already aligned as 1d data)
    # No need to align since we're using 1d data directly
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(pivot_val[i]) or np.isnan(r1_val[i]) or np.isnan(s1_val[i]) or
            np.isnan(r2_val[i]) or np.isnan(s2_val[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot or trend reverses
            if close[i] < pivot_val[i] or close[i] < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot or trend reverses
            if close[i] > pivot_val[i] or close[i] > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs weekly EMA21
            uptrend = close[i] > ema_21_1w_aligned[i]
            downtrend = close[i] < ema_21_1w_aligned[i]
            
            # Long: price breaks above R1 with volume spike + uptrend
            if (close[i] > r1_val[i] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume spike + downtrend
            elif (close[i] < s1_val[i] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals