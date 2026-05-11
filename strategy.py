#!/usr/bin/env python3
name = "6h_Premium_Discount_Order_Block"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d structure: identify swing highs/lows for order blocks
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high/low for premium/discount zones
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate 10-day ATR for volatility normalization
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align daily data to 6h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # Premium/discount zones: above/below 50% of daily range
    daily_range = daily_high_aligned - daily_low_aligned
    midpoint = daily_low_aligned + 0.5 * daily_range
    
    # Volume filter: 2-period volume spike
    vol_ma2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_filter = volume > 1.5 * vol_ma2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(midpoint[i]) or np.isnan(atr_10_aligned[i]) or np.isnan(vol_ma2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        current_price = close[i]
        atr = atr_10_aligned[i]
        
        # Define entry zones with ATR buffer
        premium_zone = midpoint[i] + 0.5 * atr
        discount_zone = midpoint[i] - 0.5 * atr
        
        if position == 0:
            # Long: price in discount zone with volume spike
            if current_price < discount_zone and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price in premium zone with volume spike
            elif current_price > premium_zone and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to midpoint or enters premium zone
            if current_price > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint or enters discount zone
            if current_price < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals