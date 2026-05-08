#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1w trend filter.
# Long when price breaks above R3 AND 1d volume > 2x 20-period average AND 1w close > 1w EMA20 (bullish trend).
# Short when price breaks below S3 AND 1d volume > 2x 20-period average AND 1w close < 1w EMA20 (bearish trend).
# Exit when price crosses back below R3 (for long) or above S3 (for short).
# Uses Camarilla pivot levels for precise entry/exit with volume confirmation and higher timeframe trend filter.
# Target: 100-180 total trades over 4 years (25-45/year) for low fee drag.

name = "4h_Camarilla_R3S3_1dVol_1wTrend"
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
    
    # 4h Camarilla levels (based on previous day's OHLC)
    # We'll use the previous day's data, so we need to get daily data first
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Camarilla: H, L, C from previous day
    # R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*1.05/2
    # PP = (H+L+C)/3, S1 = C - (H-L)*1.05/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # We'll use R3 and S3 for entries
    
    # Get previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3 and S3
    r3 = prev_close + (prev_high - prev_low) * 1.25 / 2
    s3 = prev_close - (prev_high - prev_low) * 1.25 / 2
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 1w trend filter: EMA20 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3, volume spike, 1w close > EMA20 (bullish trend)
            long_cond = (close[i] > r3_aligned[i]) and volume_filter[i] and (close_1w[i] > ema20_1w_aligned[i]) if i < len(close_1w) else False
            # Short conditions: break below S3, volume spike, 1w close < EMA20 (bearish trend)
            short_cond = (close[i] < s3_aligned[i]) and volume_filter[i] and (close_1w[i] < ema20_1w_aligned[i]) if i < len(close_1w) else False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below R3
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above S3
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals