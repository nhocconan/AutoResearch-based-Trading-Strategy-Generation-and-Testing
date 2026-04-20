#!/usr/bin/env python3
# 1h_Camarilla_Pivot_R1S1_Breakout_Volume_SessionFilter
# Hypothesis: Camarilla pivot R1/S1 breakout on 1h with volume confirmation and session filter (08-20 UTC).
# Uses 4h trend filter (price > 4h EMA200) for direction: only long in uptrend, short in downtrend.
# Volume > 1.5x 20-period average confirms breakout strength.
# Designed for low trade frequency (target 15-35/year) to avoid fee drag on 1h timeframe.
# Works in bull/bear via trend filter and bidirectional breakout logic.

name = "1h_Camarilla_Pivot_R1S1_Breakout_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter and pivot points
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate Camarilla pivots from previous 4h bar
    # Using typical price: (high + low + close) / 3
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    typical_price_4h_vals = typical_price_4h.values
    
    # Previous 4h bar's high, low, close for pivot calculation
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    r1_4h = pivot_4h + (1.1/12) * (prev_high_4h - prev_low_4h)
    s1_4h = pivot_4h - (1.1/12) * (prev_high_4h - prev_low_4h)
    r2_4h = pivot_4h + (1.1/6) * (prev_high_4h - prev_low_4h)
    s2_4h = pivot_4h - (1.1/6) * (prev_high_4h - prev_low_4h)
    
    # Align pivot levels to 1h timeframe (available after 4h bar closes)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup for indicators
    start_idx = max(200, 20)  # EMA200 warmup + volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 4h EMA200
            uptrend = close[i] > ema200_4h_aligned[i]
            downtrend = close[i] < ema200_4h_aligned[i]
            
            # Long: uptrend + price breaks above R1 + volume confirmation
            if uptrend and close[i] > r1_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + price breaks below S1 + volume confirmation
            elif downtrend and close[i] < s1_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if trend weakens or price breaks below S1 (reversal)
            if (close[i] < ema200_4h_aligned[i]) or (close[i] < s1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if trend weakens or price breaks above R1 (reversal)
            if (close[i] > ema200_4h_aligned[i]) or (close[i] > r1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals