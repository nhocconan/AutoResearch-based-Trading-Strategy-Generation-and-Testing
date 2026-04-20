#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_RangeBreakout_V1
# Hypothesis: Combines weekly pivot points with 1d volatility regime to capture breakouts from weekly ranges.
# In low volatility (ATR < 20-day MA), wait for breakout of weekly R1/S1 with volume confirmation.
# In high volatility, fade extremes (R2/S2) as mean reversion.
# Weekly context prevents whipsaws in 6h chart; volatility regime adapts to market conditions.
# Targets 20-40 trades/year by requiring weekly level + volatility regime + volume confluence.

name = "6h_1d_WeeklyPivot_RangeBreakout_V1"
timeframe = "6h"
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
    
    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Get daily data for volatility regime
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Shift by 1 to use prior week's data (no look-ahead)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Prior week's typical price and range
    typical_price_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    # Weekly pivot and support/resistance levels
    pw = typical_price_w  # weekly pivot
    r1_w = pw + range_w * 0.382  # Weekly R1 (38.2% extension)
    s1_w = pw - range_w * 0.382  # Weekly S1 (38.2% retracement)
    r2_w = pw + range_w * 0.618  # Weekly R2 (61.8% extension)
    s2_w = pw - range_w * 0.618  # Weekly S2 (61.8% retracement)
    
    # Align weekly levels to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Calculate 1d ATR for volatility regime (14-period)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder smoothing for ATR
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nanmean(arr[1:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_d = wilder_smooth(tr, 14)
    atr_ma_d = pd.Series(atr_d).rolling(window=20, min_periods=20).mean().values
    atr_d_aligned = align_htf_to_ltf(prices, df_d, atr_d)
    atr_ma_d_aligned = align_htf_to_ltf(prices, df_d, atr_ma_d)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(r2_w_aligned[i]) or np.isnan(s2_w_aligned[i]) or
            np.isnan(atr_d_aligned[i]) or np.isnan(atr_ma_d_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: low vol if ATR < 20-day MA
        low_volatility = atr_d_aligned[i] < atr_ma_d_aligned[i]
        
        if position == 0:
            if low_volatility:
                # Low volatility: wait for breakout of weekly R1/S1 with volume
                # Long breakout above R1
                if (close[i] > r1_w_aligned[i] * 1.002 and 
                    volume[i] > 1.8 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S1
                elif (close[i] < s1_w_aligned[i] * 0.998 and 
                      volume[i] > 1.8 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
            else:
                # High volatility: fade at weekly R2/S2 (mean reversion)
                # Long near S2
                if (close[i] <= s2_w_aligned[i] * 1.005 and 
                    close[i] >= s2_w_aligned[i] * 0.995):
                    signals[i] = 0.25
                    position = 1
                # Short near R2
                elif (close[i] >= r2_w_aligned[i] * 0.995 and 
                      close[i] <= r2_w_aligned[i] * 1.005):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: reverse at opposite level or volatility shift
            if low_volatility:
                # In low vol, exit if price returns inside weekly R1/S1
                if s1_w_aligned[i] <= close[i] <= r1_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In high vol, exit if price reaches opposite S2/R2 or volatility drops
                if (close[i] >= r2_w_aligned[i] * 0.995 or 
                    atr_d_aligned[i] < atr_ma_d_aligned[i] * 0.9):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse at opposite level or volatility shift
            if low_volatility:
                # In low vol, exit if price returns inside weekly R1/S1
                if s1_w_aligned[i] <= close[i] <= r1_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In high vol, exit if price reaches opposite S2/R2 or volatility drops
                if (close[i] <= s2_w_aligned[i] * 1.005 or 
                    atr_d_aligned[i] < atr_ma_d_aligned[i] * 0.9):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals