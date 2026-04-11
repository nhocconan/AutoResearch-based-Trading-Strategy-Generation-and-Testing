#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_reversion_v1
# Strategy: 4h Camarilla pivot mean reversion with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as support/resistance; mean reversion at these levels with 1d trend filter captures high-probability reversals in both bull and bear markets. Volume confirmation filters false signals. Designed for low trade frequency (<30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (using previous day's OHLC)
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We'll use H3, L3 for mean reversion entries
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels based on previous day's data
    H3 = close_1d + 1.0 * daily_range
    L3 = close_1d - 1.0 * daily_range
    
    # Align Camarilla levels to 4h timeframe (with 1-day delay since pivots based on previous day)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3, additional_delay_bars=1)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3, additional_delay_bars=1)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals at Camarilla H3/L3 levels
        # Long when price touches/slightly below L3 in uptrend
        # Short when price touches/slightly above H3 in downtrend
        near_L3 = close[i] <= L3_aligned[i] * 1.002  # Within 0.2% of L3
        near_H3 = close[i] >= H3_aligned[i] * 0.998  # Within 0.2% of H3
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Mean reversion at Camarilla levels + volume + trend alignment
        if near_L3 and vol_confirm[i] and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif near_H3 and vol_confirm[i] and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: when price moves back toward midline (close_1d) or opposite Camarilla level touched
        elif position == 1 and (close[i] >= close_1d[i] * 0.999 or close[i] >= H3_aligned[i] * 0.998):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= close_1d[i] * 1.001 or close[i] <= L3_aligned[i] * 1.002):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals