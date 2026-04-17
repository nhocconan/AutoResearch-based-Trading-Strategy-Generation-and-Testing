#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels (H3, L3, H4, L4)
    # H3 = High + 1.1*(Close - Low)/2
    # L3 = Low - 1.1*(High - Close)/2
    # H4 = High + 1.1*(Close - Low)
    # L4 = Low - 1.1*(High - Close)
    pivot_range = high_1d - low_1d
    h3 = high_1d + 1.1 * (close_1d - low_1d) / 2
    l3 = low_1d - 1.1 * (high_1d - close_1d) / 2
    h4 = high_1d + 1.1 * (close_1d - low_1d)
    l4 = low_1d - 1.1 * (high_1d - close_1d)
    
    # Align daily Camarilla levels to 6h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Weekly trend filter: price above/below weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend_up = close_1w > ema50_1w
    weekly_trend_down = close_1w < ema50_1w
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need daily pivots, weekly EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above H4 with weekly uptrend and volume
            if (close[i] > h4_aligned[i] and weekly_trend_up_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 with weekly downtrend and volume
            elif (close[i] < l4_aligned[i] and weekly_trend_down_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to L3 (mean reversion in uptrend)
            if close[i] < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to H3 (mean reversion in downtrend)
            if close[i] > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4_L4_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0