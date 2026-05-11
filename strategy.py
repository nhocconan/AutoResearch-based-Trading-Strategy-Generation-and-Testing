#!/usr/bin/env python3
"""
6h_Donchian_20_1D_Pivot_Direction_Volume
Hypothesis: 6h Donchian(20) breakouts in the direction of 1d Camarilla pivot bias (R3/S3 for continuation, R4/S4 for reversal) with volume confirmation.
Works in bull/bear markets by using pivot-based directional bias to filter breakouts. Targets 12-30 trades/year with tight entry conditions.
"""

name = "6h_Donchian_20_1D_Pivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivots (based on previous day) ---
    # Calculate pivots from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align pivots to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # --- 6h Donchian(20) ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > (1.5 * vol_ma[i])
        
        # Donchian breakout conditions
        donchian_long_break = high[i] > highest_high[i-1]
        donchian_short_break = low[i] < lowest_low[i-1]
        
        # Determine pivot-based bias
        # In bullish bias (price between S3 and R3): favor R3/S3 breakouts as continuation
        # In extreme zones (beyond R3/S3): watch R4/S4 for reversal
        price = close[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        r4 = R4_aligned[i]
        s4 = S4_aligned[i]
        
        # Bullish bias: price > S3 and < R3
        bullish_bias = (price > s3) and (price < r3)
        # Bearish bias: price < R3 and > S3
        bearish_bias = (price < r3) and (price > s3)
        # Extreme bullish: price >= R3
        extreme_bullish = price >= r3
        # Extreme bearish: price <= S3
        extreme_bearish = price <= s3
        
        # Entry logic
        long_signal = False
        short_signal = False
        
        if bullish_bias:
            # In bullish bias: long on Donchian break above R3 with volume
            long_signal = donchian_long_break and (price > r3) and vol_spike
            # Short on breakdown below S3 with volume (reversal)
            short_signal = donchian_short_break and (price < s3) and vol_spike
        elif bearish_bias:
            # In bearish bias: short on Donchian break below S3 with volume
            short_signal = donchian_short_break and (price < s3) and vol_spike
            # Long on break above R3 with volume (reversal)
            long_signal = donchian_long_break and (price > r3) and vol_spike
        elif extreme_bullish:
            # Extreme bullish: watch for reversal at R4
            short_signal = donchian_short_break and (price < r4) and vol_spike
        elif extreme_bearish:
            # Extreme bearish: watch for reversal at S4
            long_signal = donchian_long_break and (price > s4) and vol_spike
        
        # Exit logic: opposite signal or volatility extreme (touching opposite S3/R3)
        if position == 1:
            exit_signal = short_signal or (price < s3)  # Exit long if price touches S3
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            exit_signal = long_signal or (price > r3)  # Exit short if price touches R3
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals