# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R3/S3 breakouts filtered by 1d trend (EMA34) and volume spike.
# Camarilla levels from prior day define key support/resistance; breakouts capture momentum.
# 1d EMA34 filters trend direction to avoid counter-trend trades.
# Volume spike confirms breakout strength. Designed for 12-30 trades/year per symbol, works in bull/bear via trend filter.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using prior day)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    # Camarilla levels
    r3 = close_prev + range_prev * 1.1 / 4
    s3 = close_prev - range_prev * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend filter: EMA34
    ema34 = pd.Series(close_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla, EMA34, and volume EMA
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume AND above daily EMA34 (uptrend)
            if high[i] > r3_aligned[i] and volume_filter[i] and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume AND below daily EMA34 (downtrend)
            elif low[i] < s3_aligned[i] and volume_filter[i] and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR closes below daily EMA34
            if low[i] < s3_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR closes above daily EMA34
            if high[i] > r3_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals