# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Use weekly trend (EMA200) to filter 12h Camarilla R3/S3 breakouts for high-probability trades.
# Weekly EMA200 provides robust trend filter that works in both bull and bear markets.
# Volume confirmation ensures breakouts are genuine. Target: 20-40 trades/year for low friction.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate 12h Camarilla levels (using prior 12h bar)
    high_prev = df_12h['high'].shift(1).values
    low_prev = df_12h['low'].shift(1).values
    close_prev = df_12h['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev)
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev)
    
    # Align Camarilla levels to 12h timeframe (wait for 12h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA200 (200 bars), Camarilla (needs 12h bar), volume EMA (20)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA200 (uptrend) AND price breaks above Camarilla R3 with volume
            if close[i] > ema_200_aligned[i] and high[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA200 (downtrend) AND price breaks below Camarilla S3 with volume
            elif close[i] < ema_200_aligned[i] and low[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR trend turns bearish
            if low[i] < camarilla_s3_aligned[i] or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR trend turns bullish
            if high[i] > camarilla_r3_aligned[i] or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals