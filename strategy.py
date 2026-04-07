#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12-hour timeframe, use weekly Camarilla pivot levels for support/resistance with weekly trend filter.
Enter long when price touches S3 support AND weekly trend is up (price > weekly EMA50) AND volume > 1.5x 20-period average.
Enter short when price touches R3 resistance AND weekly trend is down (price < weekly EMA50) AND volume > 1.5x 20-period average.
Exit when price moves to opposite H4 level or volume drops.
Camarilla levels provide precise intraday support/resistance; weekly trend filter ensures alignment with higher timeframe.
Volume confirmation reduces false breakouts. Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Based on previous week's high, low, close
    camarilla_levels = []
    for i in range(len(w_close)):
        if i == 0:
            camarilla_levels.append({'S3': np.nan, 'S2': np.nan, 'S1': np.nan,
                                   'PP': np.nan, 'R1': np.nan, 'R2': np.nan, 'R3': np.nan})
        else:
            ph = w_high[i-1]  # previous week high
            pl = w_low[i-1]   # previous week low
            pc = w_close[i-1] # previous week close
            pp = (ph + pl + pc) / 3
            r = ph - pl
            s3 = pc - 1.1 * r / 2
            s2 = pc - 1.1 * r / 4
            s1 = pc - 1.1 * r / 6
            r1 = pc + 1.1 * r / 6
            r2 = pc + 1.1 * r / 4
            r3 = pc + 1.1 * r / 2
            camarilla_levels.append({'S3': s3, 'S2': s2, 'S1': s1,
                                   'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3})
    
    # Extract arrays
    s3_arr = np.array([x['S3'] for x in camarilla_levels])
    r3_arr = np.array([x['R3'] for x in camarilla_levels])
    h4_arr = np.array([x['R1'] for x in camarilla_levels])  # H4 equivalent
    l4_arr = np.array([x['S1'] for x in camarilla_levels])  # L4 equivalent
    
    # Weekly trend filter: price > EMA50 for uptrend
    w_close_series = pd.Series(w_close)
    w_ema50 = w_close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to 12h timeframe
    s3_12h = align_htf_to_ltf(prices, df_1w, s3_arr)
    r3_12h = align_htf_to_ltf(prices, df_1w, r3_arr)
    h4_12h = align_htf_to_ltf(prices, df_1w, h4_arr)
    l4_12h = align_htf_to_ltf(prices, df_1w, l4_arr)
    ema50_12h = align_htf_to_ltf(prices, df_1w, w_ema50)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not available
        if (np.isnan(s3_12h[i]) or np.isnan(r3_12h[i]) or 
            np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or 
            np.isnan(ema50_12h[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        weekly_uptrend = close[i] > ema50_12h[i]
        weekly_downtrend = close[i] < ema50_12h[i]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price reaches H4 level (take profit)
            if high[i] >= h4_12h[i]:
                exit_long = True
            # Exit when volume drops
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price reaches L4 level (take profit)
            if low[i] <= l4_12h[i]:
                exit_short = True
            # Exit when volume drops
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S3 support AND weekly uptrend AND volume confirmed
            long_entry = (low[i] <= s3_12h[i]) and weekly_uptrend and vol_confirmed
            
            # Short entry: price touches R3 resistance AND weekly downtrend AND volume confirmed
            short_entry = (high[i] >= r3_12h[i]) and weekly_downtrend and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals