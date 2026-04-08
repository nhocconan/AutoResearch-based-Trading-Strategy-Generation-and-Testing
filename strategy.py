#!/usr/bin/env python3
# 4h_camarilla_pivot_1d_volume_v1
# Hypothesis: Camarilla pivot levels from 1-day timeframe provide strong support/resistance.
# Long when price touches or breaks above L3 level with volume > 1.5x average and price > EMA50.
# Short when price touches or breaks below H3 level with volume > 1.5x average and price < EMA50.
# Exit when price reaches opposite H3/L3 level or returns to pivot point.
# Designed to work in both bull and bear markets by trading mean reversion at key intraday levels.
# Target: 20-35 trades/year to minimize fee drag while capturing high-probability reversals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1-day OHLC data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate typical price for pivot
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    
    # Calculate Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 and L3 are the key levels for intraday trading
    H3 = pivot + 1.1 * range_1d / 2
    L3 = pivot - 1.1 * range_1d / 2
    H4 = pivot + 1.1 * range_1d
    L4 = pivot - 1.1 * range_1d
    
    # Align 1-day levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or \
           np.isnan(pivot_aligned[i]) or np.isnan(ema50[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 level or returns to pivot
            if close[i] >= H3_aligned[i] or close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 level or returns to pivot
            if close[i] <= L3_aligned[i] or close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price touches or breaks above L3 level with volume and above EMA50
            if close[i] >= L3_aligned[i] and volume_ok and close[i] > ema50[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or breaks below H3 level with volume and below EMA50
            elif close[i] <= H3_aligned[i] and volume_ok and close[i] < ema50[i]:
                position = -1
                signals[i] = -0.25
    
    return signals