#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pullback_Bounce
Hypothesis: Combines Camarilla pivot levels from daily timeframe with pullback entries on 4h.
In trending markets, price often pulls back to key pivot levels (L3, L4, H3, H4) before continuing.
We enter long when price pulls back to L3/L4 in uptrend, short when pulls back to H3/H4 in downtrend.
Trend determined by 4h EMA(50) vs EMA(200). Works in both bull and bear markets by trading mean reversion within the trend.
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    # H4 = close + 1.1 * (high - low)
    # H3 = close + 0.55 * (high - low)
    # L3 = close - 0.55 * (high - low)
    # L4 = close - 1.1 * (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 0.55 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 0.55 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Get 4h data for trend and entry
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate trend: EMA(50) vs EMA(200) on 4h
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all signals to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or \
           np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Entry conditions: pullback to Camarilla levels in direction of trend
        if uptrend:
            # Long when price touches or crosses L3/L4 from above
            if low[i] <= camarilla_l3_aligned[i] or low[i] <= camarilla_l4_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Exit long if trend changes or price reaches H3
            elif position == 1 and (not uptrend or high[i] >= camarilla_h3_aligned[i]):
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif downtrend:
            # Short when price touches or crosses H3/H4 from below
            if high[i] >= camarilla_h3_aligned[i] or high[i] >= camarilla_h4_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Exit short if trend changes or price reaches L3
            elif position == -1 and (not downtrend or low[i] <= camarilla_l3_aligned[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # No clear trend - stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pullback_Bounce"
timeframe = "4h"
leverage = 1.0