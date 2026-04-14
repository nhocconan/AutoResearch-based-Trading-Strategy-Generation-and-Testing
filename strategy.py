#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla Pivot Reversal with 12-hour trend filter and volume confirmation
# Long when price touches L3 support with volume >1.3x average and price above 12h EMA60
# Short when price touches H3 resistance with volume >1.3x average and price below 12h EMA60
# Exit when price crosses the 4h Camarilla pivot point (PP)
# Camarilla levels: PP=(H+L+C)/3, H3=PP+(H-L)*1.1/2, L3=PP-(H-L)*1.1/2
# 12-hour EMA60 provides trend context to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot = (high_4h + low_4h + close_4h) / 3
    range_hl = high_4h - low_4h
    H3 = pivot + range_hl * 1.1 / 2
    L3 = pivot - range_hl * 1.1 / 2
    
    # Calculate 12h EMA60
    close_12h = df_12h['close'].values
    ema_60_12h = pd.Series(close_12h).ewm(span=60, min_periods=60, adjust=False).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    ema_60_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_60_12h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 60-period EMA and pivot calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_60_12h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]
        
        if position == 0:
            # Long setup: touch L3 support with volume confirmation and price above 12h EMA60
            if (price <= L3_aligned[i] * 1.002 and  # Allow small tolerance for touch
                vol_4h_current > 1.3 * vol_ma_4h_aligned[i] and  # Volume confirmation
                price > ema_60_12h_aligned[i]):                 # Price above 12h EMA60 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: touch H3 resistance with volume confirmation and price below 12h EMA60
            elif (price >= H3_aligned[i] * 0.998 and   # Allow small tolerance for touch
                  vol_4h_current > 1.3 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  price < ema_60_12h_aligned[i]):               # Price below 12h EMA60 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above pivot point
            if price > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below pivot point
            if price < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_12hEMA60_Volume"
timeframe = "4h"
leverage = 1.0