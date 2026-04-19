#!/usr/bin/env python3
"""
1d_1w_Pivot_R1S1_Breakout_Volume_Trend
Hypothesis: 1d price breaks above/below weekly R1/S1 with volume confirmation and weekly trend filter
- Weekly Pivot levels provide institutional reference points
- Volume confirmation filters for institutional participation
- Weekly EMA50 trend filter avoids counter-trend trades in reversals
- Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year)
- Works in bull/bear via trend filter and volatility-adjusted breakouts
"""

name = "1d_1w_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Previous week's OHLC for Pivot calculation
    pw_high = df_1w['high'].shift(1).values  # Previous week high
    pw_low = df_1w['low'].shift(1).values    # Previous week low
    pw_close = df_1w['close'].shift(1).values # Previous week close
    
    # Calculate Weekly Pivot levels
    pw_pivot = (pw_high + pw_low + pw_close) / 3
    pw_range = pw_high - pw_low
    
    # Weekly R1 and S1 (most significant levels)
    r1 = pw_pivot + (pw_range * 1.0)  # Standard pivot R1
    s1 = pw_pivot - (pw_range * 1.0)  # Standard pivot S1
    
    # Align Pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema50_1w_aligned[i]
        price_below_ema = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and uptrend
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or trend reverses
            if (close[i] < s1_aligned[i]) or (not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or trend reverses
            if (close[i] > r1_aligned[i]) or (not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals