#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot points with daily momentum confirmation.
- Calculate weekly pivot points (PP, R1, R2, S1, S2) from previous week OHLC
- Enter long when price crosses above weekly R1 with daily RSI > 50 (bullish momentum)
- Enter short when price crosses below weekly S1 with daily RSI < 50 (bearish momentum)
- Exit when price returns to weekly pivot point (PP)
- Uses weekly structure for direction and daily momentum for timing
- Designed for 6h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Works in bull markets (buying strength) and bear markets (selling weakness)
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
    
    # Get weekly data for pivot point calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for RSI momentum filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week OHLC
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R1 = 2*PP - Low
    # S1 = 2*PP - High
    # R2 = PP + (High - Low)
    # S2 = PP - (High - Low)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    PP = (high_weekly + low_weekly + close_weekly) / 3.0
    R1 = 2 * PP - low_weekly
    S1 = 2 * PP - high_weekly
    R2 = PP + (high_weekly - low_weekly)
    S2 = PP - (high_weekly - low_weekly)
    
    # Align weekly pivot points to 6h timeframe (use previous week's levels)
    PP_aligned = align_htf_to_ltf(prices, df_weekly, PP)
    R1_aligned = align_htf_to_ltf(prices, df_weekly, R1)
    S1_aligned = align_htf_to_ltf(prices, df_weekly, S1)
    R2_aligned = align_htf_to_ltf(prices, df_weekly, R2)
    S2_aligned = align_htf_to_ltf(prices, df_weekly, S2)
    
    # Calculate daily RSI(14) for momentum filter
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align daily RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi_values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for RSI
    
    for i in range(start_idx, n):
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        PP_val = PP_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        
        if position == 0:
            # Look for weekly pivot level breaks with daily momentum confirmation
            # Long: price crosses above weekly R1 + daily RSI > 50 (bullish momentum)
            if price > R1_val and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly S1 + daily RSI < 50 (bearish momentum)
            elif price < S1_val and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to weekly pivot point (mean reversion to center)
            if price < PP_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to weekly pivot point (mean reversion to center)
            if price > PP_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_DailyRSI50"
timeframe = "6h"
leverage = 1.0