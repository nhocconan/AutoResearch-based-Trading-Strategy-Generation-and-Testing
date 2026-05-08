# [139570] 1d_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: On daily timeframe, price breaking above/below S1/R1 with trend confirmation and volume spike captures institutional breakouts.
# Uses 1d trend (price vs EMA50) and volume confirmation to filter false breakouts. Designed for 30-100 trades over 4 years.
# Works in bull/bear: trend filter ensures alignment with higher timeframe bias.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once (same timeframe, but needed for prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate prior day's OHLC for Camarilla levels
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels using prior day's data
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    range_ = high_1d_prev - low_1d_prev
    r1 = close_1d_prev + (range_ * 1.1 / 12)
    s1 = close_1d_prev - (range_ * 1.1 / 12)
    r3 = close_1d_prev + (range_ * 1.1 / 4)
    s3 = close_1d_prev - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above S1 + uptrend + volume spike
            if (close[i] > s1_val and 
                close[i] > ema50_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R1 + downtrend + volume spike
            elif (close[i] < r1_val and 
                  close[i] < ema50_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR trend turns down
            if (close[i] < s3_val or close[i] < ema50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR trend turns up
            if (close[i] > r3_val or close[i] > ema50_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals