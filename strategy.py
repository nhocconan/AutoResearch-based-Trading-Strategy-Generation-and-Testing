#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h EMA50 Trend Filter and Volume Confirmation
# Uses 4h Camarilla pivot levels (R3/S3 for continuation, R4/S4 for breakout) from 4h timeframe
# Entry logic: Break above R3 with volume spike in uptrend (price > 4h EMA50) for long
#              Break below S3 with volume spike in downtrend (price < 4h EMA50) for short
# Works in both bull and bear markets by trading with the 4h trend
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Discrete sizing 0.20 balances profit potential and fee drag

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formulas: Pivot = (H+L+C)/3, Range = H-L
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Resistance levels: R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    r3_4h = close_4h + (range_4h * 1.1 / 4.0)
    r4_4h = close_4h + (range_4h * 1.1 / 2.0)
    s3_4h = close_4h - (range_4h * 1.1 / 4.0)
    s4_4h = close_4h - (range_4h * 1.1 / 2.0)
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above R3 AND price > 4h EMA50 (uptrend) AND volume spike
            if (close[i] > r3_4h_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Break below S3 AND price < 4h EMA50 (downtrend) AND volume spike
            elif (close[i] < s3_4h_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 4h EMA50 (trend change) OR break below S4 (reversal)
            if (close[i] < ema_50_4h_aligned[i] or 
                close[i] < s4_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close above 4h EMA50 (trend change) OR break above R4 (reversal)
            if (close[i] > ema_50_4h_aligned[i] or 
                close[i] > r4_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals