#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot with 1-week trend filter (EMA50) and volume confirmation
# Long when price breaks above Camarilla H3 level AND price > weekly EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Camarilla L3 level AND price < weekly EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back to Camarilla Pivot level (central)
# Camarilla levels derived from prior day's range. Weekly EMA50 filters trend direction.
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla calculation (needs prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from prior day's range
    # H4 = C + 1.5*(H-L), H3 = C + 1.25*(H-L), H2 = C + 1.083*(H-L), H1 = C + 1.042*(H-L)
    # Pivot = (H+L+C)/3, L1 = C - 1.042*(H-L), L2 = C - 1.083*(H-L), L3 = C - 1.25*(H-L), L4 = C - 1.5*(H-L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate for each 4h bar using prior day's data (shifted by 1 day)
    # Since we're on 4h timeframe, we need to use prior day's OHLC for current day's levels
    # Shift daily data by 1 to get prior day's values
    if len(high_1d) > 1:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        # Set first day's values to zero (will be handled by NaN check)
        prev_high[0] = 0
        prev_low[0] = 0
        prev_close[0] = 0
    else:
        prev_high = high_1d
        prev_low = low_1d
        prev_close = close_1d
    
    # Calculate Camarilla levels
    H_L = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3.0
    H3 = prev_close + 1.25 * H_L
    L3 = prev_close - 1.25 * H_L
    
    # Align to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
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
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN or zero (invalid)
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg[i]) or H3_aligned[i] == 0 or L3_aligned[i] == 0):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above H3 + above weekly EMA50 + volume confirmation
            if (price > H3_aligned[i] and price > ema50_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below L3 + below weekly EMA50 + volume confirmation
            elif (price < L3_aligned[i] and price < ema50_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back to pivot level
            if price < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back to pivot level
            if price > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_1wEMA50_Volume"
timeframe = "4h"
leverage = 1.0