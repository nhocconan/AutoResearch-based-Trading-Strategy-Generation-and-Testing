# 4h_Camarilla_R1S1_12hEMA50_Volume
# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 AND price > EMA50(12h) AND volume > 1.5x 20-period average.
# Short when price breaks below S1 AND price < EMA50(12h) AND volume > 1.5x 20-period average.
# Exit when price crosses back below R1 (long) or above S1 (short).
# Uses tighter inner Camarilla levels (R1/S1) for higher probability entries.
# EMA50 on 12h filters trend direction. Volume confirms institutional participation.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 12h data for Camarilla pivot and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot
    typical_price = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels (R1, S1 - inner levels for higher probability)
    camarilla_r1 = typical_price + (range_12h * 1.1 / 12)
    camarilla_s1 = typical_price - (range_12h * 1.1 / 12)
    
    # EMA50 on 12h close
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R1, price > EMA50, volume filter
            long_cond = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_50_aligned[i]) and volume_filter[i]
            # Short conditions: break below S1, price < EMA50, volume filter
            short_cond = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below R1
            if close[i] < camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above S1
            if close[i] > camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals