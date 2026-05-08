#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above R1 AND price > EMA200(1d) AND volume > 1.3x 20-period average.
# Short when price breaks below S1 AND price < EMA200(1d) AND volume > 1.3x 20-period average.
# Exit when price crosses back below R1 (long) or above S1 (short).
# Uses tighter inner Camarilla levels (R1/S1) for higher probability entries.
# EMA200 on 1d filters long-term trend. Volume confirms institutional participation.
# Target: 50-100 total trades over 4 years (12-25/year) to avoid excessive trading and fee drag.

name = "4h_Camarilla_R1S1_1dEMA200_Volume"
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
    
    # 4h volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    # 1d data for Camarilla pivot and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (R1, S1 - inner levels for higher probability)
    camarilla_r1 = typical_price + (range_1d * 1.1 / 12)
    camarilla_s1 = typical_price - (range_1d * 1.1 / 12)
    
    # EMA200 on 1d close
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R1, price > EMA200, volume filter
            long_cond = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_200_aligned[i]) and volume_filter[i]
            # Short conditions: break below S1, price < EMA200, volume filter
            short_cond = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_200_aligned[i]) and volume_filter[i]
            
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