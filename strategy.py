#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d data for weekly Camarilla pivot levels (using 1d OHLC to approximate weekly structure).
- Entry: Long when price breaks above prior 1d H4 (Camarilla H4) AND 1d close > weekly pivot (bullish bias)
         Short when price breaks below prior 1d L4 (Camarilla L4) AND 1d close < weekly pivot (bearish bias)
- Volume confirmation: volume > 1.5 * volume MA(20) to avoid breakout failures
- Exit: Close-based reversal - exit long when price crosses below prior 1d L3,
        exit short when price crosses above prior 1d H3 (using Camarilla levels for dynamic stop)
- Signal size: 0.25 discrete to balance return and drawdown.
Uses weekly pivot bias from 1d data (proven edge from DB top performers) for BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot approximation and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d weekly pivot approximation: (weekly high + weekly low + weekly close) / 3
    # Using 1d data: approximate weekly by taking last 5 days (1 trading week)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot approximation using last 5 days
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate prior 1d Camarilla levels (H4, L4, H3, L3) for entries and exits
    rang = high_1d - low_1d
    camarilla_h4 = close_1d + rang * 1.1 / 2  # H4 level
    camarilla_l4 = close_1d - rang * 1.1 / 2  # L4 level
    camarilla_h3 = close_1d + rang * 1.1 / 4  # H3 level
    camarilla_l3 = close_1d - rang * 1.1 / 4  # L3 level
    
    # Align HTF indicators to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 5)  # Need enough bars for volume MA and weekly calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above prior 1d H4 AND 1d close > weekly pivot (bullish bias) AND volume confirmed
            if curr_close > camarilla_h4_aligned[i] and close_1d[-1] > weekly_pivot_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 1d L4 AND 1d close < weekly pivot (bearish bias) AND volume confirmed
            elif curr_close < camarilla_l4_aligned[i] and close_1d[-1] < weekly_pivot_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 1d L3 (dynamic stop)
            if curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 1d H3 (dynamic stop)
            if curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_Camarilla_H4L4_Entry_v1"
timeframe = "6h"
leverage = 1.0