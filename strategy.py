#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: For 12h timeframe, use daily Camarilla R1/S1 levels for breakout entries.
# In trending markets (1d EMA50), price breaks R1/S1 and continues; in ranging markets, fewer triggers.
# 1d trend filter avoids counter-trend trades. Volume confirmation reduces false breakouts.
# Designed for 12h to capture multi-day moves with low frequency (~12-37/year).
# Works in bull (breakouts continue) and bear (breakdowns continue) via trend filter.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    R1 = typical_price + (range_val * 1.1 / 4)
    S1 = typical_price - (range_val * 1.1 / 4)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    R1_prev = R1.shift(1).values
    S1_prev = S1.shift(1).values
    
    # 1d trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align daily data to 12h
    R1_prev_aligned = align_htf_to_ltf(prices, df_1d, R1_prev)
    S1_prev_aligned = align_htf_to_ltf(prices, df_1d, S1_prev)
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 2-period (1-day) average on 12h
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_prev_aligned[i]) or np.isnan(S1_prev_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: break above R1 with 1d uptrend and volume
            if (close[i] > R1_prev_aligned[i] and 
                trend_1d_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with 1d downtrend and volume
            elif (close[i] < S1_prev_aligned[i] and 
                  trend_1d_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price returns to typical price or trend fails
            typical_price_aligned = ((df_1d['high'] + df_1d['low'] + df_1d['close']) / 3).shift(1).values
            typical_price_aligned = align_htf_to_ltf(prices, df_1d, typical_price_aligned)
            if (close[i] < typical_price_aligned[i] or 
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price returns to typical price or trend fails
            typical_price_aligned = ((df_1d['high'] + df_1d['low'] + df_1d['close']) / 3).shift(1).values
            typical_price_aligned = align_htf_to_ltf(prices, df_1d, typical_price_aligned)
            if (close[i] > typical_price_aligned[i] or 
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals