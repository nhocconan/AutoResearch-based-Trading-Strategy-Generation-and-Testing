#!/usr/bin/env python3
"""
1h_pivot_reversal_4h1d_v1
Hypothesis: On 1-hour timeframe, use mean-reversion at pivot points with 4h trend filter and daily volume confirmation.
Long when price bounces off weekly pivot support with 4h EMA trending up and daily volume above average.
Short when price reverses at weekly pivot resistance with 4h EMA trending down and daily volume above average.
Exit when price reaches opposite pivot level or midpoint.
Designed for 15-35 trades/year by combining pivot levels (institutional levels) with trend and volume filters.
Works in bull/bear markets as pivots adapt to price action and filters avoid chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_pivot_reversal_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Determine 4h trend direction (using EMA slope)
    trend_up = np.zeros(len(ema_34_4h_aligned), dtype=bool)
    trend_down = np.zeros(len(ema_34_4h_aligned), dtype=bool)
    for i in range(1, len(ema_34_4h_aligned)):
        if not np.isnan(ema_34_4h_aligned[i]) and not np.isnan(ema_34_4h_aligned[i-1]):
            trend_up[i] = ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]
            trend_down[i] = ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We'll approximate weekly from daily data: use prior 5 trading days
    df_1d_for_pivot = df_1d.copy()
    if len(df_1d_for_pivot) < 5:
        return np.zeros(n)
    
    # Calculate rolling weekly high/low/close from daily data
    weekly_high = pd.Series(df_1d_for_pivot['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d_for_pivot['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d_for_pivot['close']).rolling(window=5, min_periods=5).last().values
    
    # Standard pivot point formula: (H + L + C) / 3
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    # Support 1: (2 * P) - H
    support_1 = (2 * pivot_point) - weekly_high
    # Resistance 1: (2 * P) - L
    resistance_1 = (2 * pivot_point) - weekly_low
    
    # Align pivot levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, pivot_point)
    support_1_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, support_1)
    resistance_1_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, resistance_1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(support_1_aligned[i]) or 
            np.isnan(resistance_1_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1h volume > daily average volume
        vol_ok = volume[i] > vol_avg_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches pivot point or resistance
            if close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price reaches pivot point or support
            if close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter with volume confirmation and 4h trend alignment
            if vol_ok:
                # Long: price bounces off support with 4h uptrend
                if (close[i] > support_1_aligned[i] and close[i-1] <= support_1_aligned[i-1] and 
                    trend_up[i]):
                    position = 1
                    signals[i] = 0.20
                # Short: price reverses at resistance with 4h downtrend
                elif (close[i] < resistance_1_aligned[i] and close[i-1] >= resistance_1_aligned[i-1] and 
                      trend_down[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals