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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    # For weekly pivot, we need previous week's data
    # We'll calculate pivot for each week and use it for the entire week
    
    # Create arrays for weekly values
    week_high = df_1w['high'].values
    week_low = df_1w['low'].values
    week_close = df_1w['close'].values
    
    # Calculate weekly pivot and levels
    pivot = (week_high + week_low + week_close) / 3
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    r3 = week_high + 2 * (pivot - week_low)
    s3 = week_low - 2 * (week_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 6-period RSI for momentum confirmation (6h timeframe)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=6, min_periods=6).mean()
    avg_loss = loss.rolling(window=6, min_periods=6).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    
    # Calculate volume ratio (current volume vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = vol_ratio[i]
        
        # Volume filter: require above-average volume
        vol_filter = vol > 1.2
        
        if position == 0:
            # Long setup: price crosses above S1 with volume confirmation
            # AND price is below pivot (avoiding buying too high)
            if (price > s1_aligned[i] and 
                price < pivot_aligned[i] and 
                vol_filter and
                rsi[i] < 60):  # Not overbought
                position = 1
                signals[i] = position_size
            # Short setup: price crosses below R1 with volume confirmation
            # AND price is above pivot (avoiding selling too low)
            elif (price < r1_aligned[i] and 
                  price > pivot_aligned[i] and 
                  vol_filter and
                  rsi[i] > 40):  # Not oversold
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below S1 or reaches R1
            if price < s1_aligned[i] or price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above R1 or reaches S1
            if price > r1_aligned[i] or price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wPivot_S1R1_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0