#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with weekly trend filter (EMA50) and volume confirmation
# Long when price breaks above H3 level AND price > weekly EMA50 AND volume > 2x 20-period average
# Short when price breaks below L3 level AND price < weekly EMA50 AND volume > 2x 20-period average
# Exit when price crosses back inside the Camarilla range (L3 to H3)
# Camarilla levels provide institutional support/resistance; weekly EMA50 filters trend direction
# Volume confirmation ensures breakout authenticity. Target: 60-120 total trades over 4 years (15-30/year)

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
    
    # Calculate Camarilla pivot levels from previous day (H3, L3)
    # Need daily high/low/close for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (H3, L3)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h3 = close_1d + (range_1d * 1.1 / 2)  # H3 = Close + 1.1*(Range)/2
    l3 = close_1d - (range_1d * 1.1 / 2)  # L3 = Close - 1.1*(Range)/2
    
    # Align H3/L3 to 12h timeframe (using previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 2.0
        
        if position == 0:
            # Long setup: breakout above H3 + above weekly EMA50 + volume confirmation
            if (price > h3_aligned[i] and price > ema50_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below L3 + below weekly EMA50 + volume confirmation
            elif (price < l3_aligned[i] and price < ema50_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below L3 (opposite level)
            if price < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above H3 (opposite level)
            if price > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0