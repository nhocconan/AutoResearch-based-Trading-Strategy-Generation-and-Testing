#!/usr/bin/env python3
"""
12h_DailyCamarilla_LongOnly - Uses daily Camarilla pivot levels (S3/S4 for long, R3/R4 for short)
with volume confirmation and 12h EMA trend filter. Enters long when price touches S3/S4 with
volume > 1.5x average and price above 12h EMA50. Enters short when price touches R3/R4 with
volume > 1.5x average and price below 12h EMA50. Exits on opposite signal or when price
returns to the daily pivot (mean reversion). Designed for 12h timeframe to work in both
bull and bear markets by fading extremes and trading with trend filter.
"""

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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's OHLC (to avoid look-ahead)
    phigh = df_1d['high'].shift(1).values
    plow = df_1d['low'].shift(1).values
    pclose = df_1d['close'].shift(1).values
    popen = df_1d['open'].shift(1).values
    
    # Camarilla levels (based on prior day's range)
    range_val = phigh - plow
    # S1, S2, S3, S4 (support levels)
    s1 = pclose - range_val * 1.1 / 12
    s2 = pclose - range_val * 1.1 / 6
    s3 = pclose - range_val * 1.1 / 4
    s4 = pclose - range_val * 1.1 / 2
    # R1, R2, R3, R4 (resistance levels)
    r1 = pclose + range_val * 1.1 / 12
    r2 = pclose + range_val * 1.1 / 6
    r3 = pclose + range_val * 1.1 / 4
    r4 = pclose + range_val * 1.1 / 2
    # Pivot point
    pp = (phigh + plow + pclose) / 3
    
    # Align daily levels to 12h timeframe
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp)
    
    # 12h EMA50 for trend filter (using prior closes)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or np.isnan(r3_12h[i]) or 
            np.isnan(r4_12h[i]) or np.isnan(pp_12h[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price touches S3 or S4 with volume confirmation and above EMA50
            if (price <= s3_12h[i] or price <= s4_12h[i]) and vol > 1.5 * vol_ma and price > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches R3 or R4 with volume confirmation and below EMA50
            elif (price >= r3_12h[i] or price >= r4_12h[i]) and vol > 1.5 * vol_ma and price < ema_50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to daily pivot or touches resistance (mean reversion)
            if price >= pp_12h[i] or price >= r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to daily pivot or touches support (mean reversion)
            if price <= pp_12h[i] or price <= s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyCamarilla_LongOnly"
timeframe = "12h"
leverage = 1.0