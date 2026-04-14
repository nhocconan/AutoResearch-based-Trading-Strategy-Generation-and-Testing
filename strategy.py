#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with 1-day trend filter (EMA50) and volume confirmation
# Long when price breaks above Camarilla H4 level AND price > daily EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Camarilla L4 level AND price < daily EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back through Camarilla L3/H3 levels
# Uses proven Camarilla pivot structure with trend and volume filters to reduce false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily data
    # Pivot point = (H + L + C) / 3
    # H4 = C + (H - L) * 1.1/2
    # L4 = C - (H - L) * 1.1/2
    # H3 = C + (H - L) * 1.1/4
    # L3 = C - (H - L) * 1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    H4 = close_1d + range_hl * 1.1 / 2.0
    L4 = close_1d - range_hl * 1.1 / 2.0
    H3 = close_1d + range_hl * 1.1 / 4.0
    L3 = close_1d - range_hl * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # Need EMA50 period
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above H4 + above daily EMA50 + volume confirmation
            if (price > H4_aligned[i] and price > ema50_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below L4 + below daily EMA50 + volume confirmation
            elif (price < L4_aligned[i] and price < ema50_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below L3 (opposite side)
            if price < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above H3 (opposite side)
            if price > H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0