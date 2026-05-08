#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = prices['high'].shift(1)
    prev_low = prices['low'].shift(1)
    prev_close = prices['close'].shift(1)
    
    # Camarilla pivot formula
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance and Support levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Volume confirmation
    vol_ma20 = pd.Series(prices['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema_12h_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, 12h uptrend, volume spike
            long_cond = (prices['close'].iloc[i] > r3[i] and 
                        ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below S3, 12h downtrend, volume spike
            short_cond = (prices['close'].iloc[i] < s3[i] and 
                         ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S3
            if prices['close'].iloc[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3
            if prices['close'].iloc[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals