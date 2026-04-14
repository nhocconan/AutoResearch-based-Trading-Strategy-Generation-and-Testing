#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with 1-week trend filter (EMA50) and volume confirmation
# Long when price breaks above H4 Camarilla level AND price > weekly EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below L4 Camarilla level AND price < weekly EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Camarilla H4-L4 range
# This captures strong trending moves from key intraday pivot levels with volume confirmation while avoiding counter-trend trades
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily data
    # Using previous day's OHLC for today's levels (standard practice)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4, H3, L3, L4
    # H4 = close + 1.5 * range
    # L4 = close - 1.5 * range
    h4_1d = close_1d + 1.5 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Align daily Camarilla levels to 4h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # Need enough data for weekly EMA50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above H4 Camarilla level + above weekly EMA50 + volume confirmation
            if (price > h4_1d_aligned[i] and price > ema50_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below L4 Camarilla level + below weekly EMA50 + volume confirmation
            elif (price < l4_1d_aligned[i] and price < ema50_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below L4 Camarilla level (opposite side of range)
            if price < l4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above H4 Camarilla level (opposite side of range)
            if price > h4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1wCamarilla_EMA50_Volume"
timeframe = "4h"
leverage = 1.0