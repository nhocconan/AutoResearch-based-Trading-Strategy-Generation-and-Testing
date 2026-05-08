#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Using typical price of previous day: (H + L + C) / 3
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # avoid NaN on first element
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    typical_price = (prev_high + prev_low + prev_close) / 3
    
    # Camarilla levels: R3, R2, R1, S1, S2, S3
    # Range = high - low
    range_1d = prev_high - prev_low
    R3 = typical_price + range_1d * 1.1 / 2
    R2 = typical_price + range_1d * 1.1 / 4
    R1 = typical_price + range_1d * 1.1 / 6
    S1 = typical_price - range_1d * 1.1 / 6
    S2 = typical_price - range_1d * 1.1 / 4
    S3 = typical_price - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d trend filter: EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R3 + uptrend + volume confirmation
            if (close[i] > R3_aligned[i] and
                close[i] > ema_34_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 + downtrend + volume confirmation
            elif (close[i] < S3_aligned[i] and
                  close[i] < ema_34_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below R1 or trend reversal
            if (close[i] < R1_aligned[i] or
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above S1 or trend reversal
            if (close[i] > S1_aligned[i] or
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals