#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend and volume confirmation.
# Uses 1d Camarilla levels (R3/S3 for reversal, R4/S4 for breakout) and 1d EMA trend filter.
# In bull markets: buy R3/S3 bounce in uptrend, sell R4/S4 breakout in uptrend.
# In bear markets: sell R3/S3 rejection in downtrend, buy R4/S4 breakdown in downtrend.
# Target: 12-37 trades/year to avoid fee drag.
name = "6h_Camarilla_R3S3_R4S4_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    pivot = (H + L + C) / 3.0
    rng = H - L
    r4 = C + rng * 1.1 / 2.0
    r3 = C + rng * 1.1 / 4.0
    s3 = C - rng * 1.1 / 4.0
    s4 = C - rng * 1.1 / 2.0
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 6h timeframe (use previous day's values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data for pivots/EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: R3/S3 bounce in uptrend OR R4/S4 breakout in uptrend + volume
            if ((price >= r3_aligned[i] and price <= s3_aligned[i]) and 
                price > ema_34_1d_aligned[i] and vol_confirm[i]):
                # In the range between S3 and R3, go long in uptrend
                signals[i] = 0.25
                position = 1
            elif (price > r4_aligned[i] and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                # Breakout above R4 in uptrend
                signals[i] = 0.25
                position = 1
            elif (price < s4_aligned[i] and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                # Breakdown below S4 in downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R3 or trend reverses
            if price < r3_aligned[i] or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S3 or trend reverses
            if price > s3_aligned[i] or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals