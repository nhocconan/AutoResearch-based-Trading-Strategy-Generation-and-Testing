#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Trend_4h
Hypothesis: Combines 1d Camarilla pivot levels with 4h EMA trend and volume confirmation.
Enters long when 4h close > H4 (1d) and 4h EMA20 > EMA50 with volume expansion.
Enters short when 4h close < L4 (1d) and 4h EMA20 < EMA50 with volume expansion.
Uses discrete position sizing (0.25) to limit turnover. Designed for 4h timeframe
to target 20-50 trades/year (80-200 total over 4 years). Works in bull markets via
trend-following breakouts and in bear markets via mean-reversion off extreme levels.
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous 1d bar
    hl_range = high_1d - low_1d
    H4 = close_1d + 1.125 * hl_range
    L4 = close_1d - 1.125 * hl_range
    
    # Calculate 4h EMA20 and EMA50 for trend
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 20-period volume average on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all signals to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(ema20[i]) or
            np.isnan(ema50[i])):
            signals[i] = 0.0
            continue
        
        # Volume expansion: current 4h volume > 1.5x 1d volume MA
        volume_expansion = volume[i] > (vol_ma_20_1d_aligned[i] * 1.5)
        
        # Trend condition: EMA20 > EMA50 for long, EMA20 < EMA50 for short
        uptrend = ema20[i] > ema50[i]
        downtrend = ema20[i] < ema50[i]
        
        # Entry conditions: price breaks H4/L4 with volume expansion and trend alignment
        long_entry = (close[i] > H4_aligned[i]) and volume_expansion and uptrend
        short_entry = (close[i] < L4_aligned[i]) and volume_expansion and downtrend
        
        # Exit conditions: price returns to opposite Camarilla level (H4 for shorts, L4 for longs)
        exit_long = position == 1 and close[i] <= L4_aligned[i]
        exit_short = position == -1 and close[i] >= H4_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Breakout_Trend_4h"
timeframe = "4h"
leverage = 1.0