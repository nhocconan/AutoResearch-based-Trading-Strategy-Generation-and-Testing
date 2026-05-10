#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_Volume
Hypothesis: Uses daily Camarilla pivot levels (R4/S4) for breakout entries with daily trend filter and volume confirmation. 
In strong trends, price breaks R4/S4 and continues; in ranging markets, fewer triggers reduce whipsaw. 
Daily trend filter avoids counter-trend trades. Designed for 6h timeframe to capture multi-day moves with low frequency.
Target: 12-30 trades/year per symbol.
"""

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_val = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R4 = typical_price + (range_val * 1.1 / 2)
    R3 = typical_price + (range_val * 1.1 / 4)
    S3 = typical_price - (range_val * 1.1 / 4)
    S4 = typical_price - (range_val * 1.1 / 2)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    R4_prev = R4.shift(1).values
    S4_prev = S4.shift(1).values
    
    # Daily trend: EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align daily data to 6h
    R4_prev_aligned = align_htf_to_ltf(prices, df_1d, R4_prev)
    S4_prev_aligned = align_htf_to_ltf(prices, df_1d, S4_prev)
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 24-period (4-day) average on 6h
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R4_prev_aligned[i]) or np.isnan(S4_prev_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: break above R4 with daily uptrend and volume
            if (close[i] > R4_prev_aligned[i] and 
                trend_1d_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: break below S4 with daily downtrend and volume
            elif (close[i] < S4_prev_aligned[i] and 
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