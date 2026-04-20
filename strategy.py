#!/usr/bin/env python3
# 1d_Keltner_Channel_Breakout_With_1w_Trend_Filter
# Hypothesis: Price breaking above/below Keltner Channel (20, 2.0) on 1d timeframe, filtered by 1w EMA50 trend direction.
# In bull markets (close > 1w EMA50): long when price > upper KC (EMA20 + 2*ATR).
# In bear markets (close < 1w EMA50): short when price < lower KC (EMA20 - 2*ATR).
# Keltner Channel adapts to volatility, reducing false breakouts in ranging markets.
# EMA50 trend filter ensures alignment with higher timeframe momentum.
# Target: 20-80 total trades over 4 years (5-20/year) to minimize fee drag.

name = "1d_Keltner_Channel_Breakout_With_1w_Trend_Filter"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Keltner Channel on 1d data
    kc_period = 20
    kc_multiplier = 2.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr = np.full_like(high, np.nan)
    if len(high) >= kc_period:
        atr[kc_period] = np.nanmean(tr[1:kc_period+1])
        for i in range(kc_period + 1, len(high)):
            atr[i] = (atr[i-1] * (kc_period - 1) + tr[i]) / kc_period
    
    # EMA20 for middle line
    ema20 = np.full_like(close, np.nan)
    if len(close) >= kc_period:
        ema20[kc_period-1] = np.mean(close[:kc_period])
        for i in range(kc_period, len(close)):
            ema20[i] = (close[i] * 2 + ema20[i-1] * (kc_period - 1)) / (kc_period + 1)
    
    # Upper and Lower Keltner Channel
    upper_kc = np.full_like(close, np.nan)
    lower_kc = np.full_like(close, np.nan)
    valid_kc = ~np.isnan(ema20) & ~np.isnan(atr)
    upper_kc[valid_kc] = ema20[valid_kc] + (kc_multiplier * atr[valid_kc])
    lower_kc[valid_kc] = ema20[valid_kc] - (kc_multiplier * atr[valid_kc])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(upper_kc[i]) or 
            np.isnan(lower_kc[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1w EMA50
            uptrend = close[i] > ema50_1w_aligned[i]
            downtrend = close[i] < ema50_1w_aligned[i]
            
            # Long: uptrend + price > upper KC
            if uptrend and close[i] > upper_kc[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price < lower KC
            elif downtrend and close[i] < lower_kc[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price closes below middle KC or trend reverses
            if close[i] < ema20[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price closes above middle KC or trend reverses
            if close[i] > ema20[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals