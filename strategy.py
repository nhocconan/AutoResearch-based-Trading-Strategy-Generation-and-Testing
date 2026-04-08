#!/usr/bin/env python3
"""
4h_4h_1d_camarilla_pullback_v1
Hypothesis: Trade pullbacks to daily Camarilla pivot levels in the direction of 4h trend.
- Only trade in direction of 4h EMA trend (above/below EMA50)
- Long: 4h bullish + price pulls back to daily pivot then closes above it
- Short: 4h bearish + price pulls back to daily pivot then closes below it
- Exit on opposite pullback or 4h trend reversal
- Target: 20-30 trades/year to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_4h_1d_camarilla_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h EMA for trend
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    bullish = close > ema_50
    bearish = close < ema_50
    
    # Daily data for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla pivot (H4/L4 levels)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Camarilla levels: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    camarilla_h4 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 2
    camarilla_l4 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 2
    
    # Align daily Camarilla levels to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: pullback below L4 or 4h turns bearish
            if close[i] < camarilla_l4_aligned[i] or bearish[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: pullback above H4 or 4h turns bullish
            if close[i] > camarilla_h4_aligned[i] or bullish[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: 4h bullish + pullback to L4 then close above
            if (bullish[i] and 
                close[i-1] <= camarilla_l4_aligned[i-1] and close[i] > camarilla_l4_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: 4h bearish + pullback to H4 then close below
            elif (bearish[i] and 
                  close[i-1] >= camarilla_h4_aligned[i-1] and close[i] < camarilla_h4_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals