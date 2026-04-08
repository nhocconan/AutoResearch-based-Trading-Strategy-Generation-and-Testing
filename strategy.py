#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_breakout_1w_trend_volume_v3"
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
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (using previous week's data)
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    
    # Weekly trend: EMA21
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to daily
    pivot_d = align_htf_to_ltf(prices, df_1w, pivot)
    r1_d = align_htf_to_ltf(prices, df_1w, r1)
    s1_d = align_htf_to_ltf(prices, df_1w, s1)
    r2_d = align_htf_to_ltf(prices, df_1w, r2)
    s2_d = align_htf_to_ltf(prices, df_1w, s2)
    ema_21_d = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Daily volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_d[i]) or np.isnan(r1_d[i]) or np.isnan(s1_d[i]) or
            np.isnan(r2_d[i]) or np.isnan(s2_d[i]) or np.isnan(ema_21_d[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S1 or weekly trend turns bearish
            if close[i] < s1_d[i] or close[i] < ema_21_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R1 or weekly trend turns bullish
            if close[i] > r1_d[i] or close[i] > ema_21_d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price > R1 + above weekly EMA + volume
            if (close[i] > r1_d[i] and 
                close[i] > ema_21_d[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < S1 + below weekly EMA + volume
            elif (close[i] < s1_d[i] and 
                  close[i] < ema_21_d[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals