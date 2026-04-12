#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Reversal
Hypothesis: Trade reversals at Camarilla pivot levels (H4/L4) on 1d timeframe with volume confirmation and weekly trend filter.
Long when price touches L4 with rising volume in weekly uptrend, short when price touches H4 with rising volume in weekly downtrend.
Designed for 10-25 trades/year with clear reversal logic that works in bull (buy dips) and bear (sell rallies) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 for weekly trend
    close_1w_series = pd.Series(close_1w)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = ema20_1w > np.roll(ema20_1w, 1)
    weekly_downtrend = ema20_1w < np.roll(ema20_1w, 1)
    # Handle first value
    weekly_uptrend[0] = False
    weekly_downtrend[0] = False
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Calculate Camarilla levels for current day using previous day's OHLC
        if i == 0:
            # Cannot calculate for first bar
            signals[i] = 0.0
            continue
            
        # Previous day's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla levels
        range_val = ph - pl
        if range_val <= 0:
            # Skip if no range
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
            
        h4 = pc + 1.1 * range_val * 1.1 / 2  # H4 = C + 1.1*(H-L)*1.1/2
        l4 = pc - 1.1 * range_val * 1.1 / 2  # L4 = C - 1.1*(H-L)*1.1/2
        h3 = pc + 1.1 * range_val / 2        # H3 = C + 1.1*(H-L)/2
        l3 = pc - 1.1 * range_val / 2        # L3 = C - 1.1*(H-L)/2
        
        # Volume filter: current volume > 1.5x average of last 20 days
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        strong_volume = volume[i] > (vol_ma * 1.5)
        
        # Price touch tolerance (0.1% of price)
        tolerance = close[i] * 0.001
        
        # Long: price touches L4 with strong volume in weekly uptrend
        long_signal = (abs(close[i] - l4) <= tolerance and 
                      strong_volume and 
                      weekly_uptrend_aligned[i])
        
        # Short: price touches H4 with strong volume in weekly downtrend
        short_signal = (abs(close[i] - h4) <= tolerance and 
                       strong_volume and 
                       weekly_downtrend_aligned[i])
        
        # Exit: price reaches H3/L3 or opposite Camarilla level
        exit_long = (position == 1 and 
                    (close[i] >= h3 or close[i] <= l4))
        exit_short = (position == -1 and 
                     (close[i] <= l3 or close[i] >= h4))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals