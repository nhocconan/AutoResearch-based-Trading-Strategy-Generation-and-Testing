# [EXPERIMENT #153998] 1d_WeeklyPivot_Breakout_1wTrend_Volume
# Hypothesis: Use weekly Camarilla pivot levels as support/resistance, enter on daily breakouts
# with volume confirmation, filtered by 1-week trend direction. Weekly pivots provide robust
# S/R levels that work in both bull and bear markets. Volume filter reduces false breakouts.
# Trend filter ensures alignment with higher timeframe momentum. Target: 10-25 trades/year.

#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots (using previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's range
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla R3 and S3 (most commonly used breakout levels)
    camarilla_r3 = close_1w + (range_1w * 1.1 / 2)
    camarilla_s3 = close_1w - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to daily timeframe
    r3_daily = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_daily = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # 1-week EMA for trend filter
    close_series = pd.Series(close)
    ema_1w = close_series.ewm(span=10, min_periods=10).mean().values  # 10-week EMA on daily
    
    # Volume filter: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_daily[i]) or np.isnan(s3_daily[i]) or 
            np.isnan(ema_1w[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 AND above weekly EMA (uptrend) AND volume spike
            if close[i] > r3_daily[i] and close[i] > ema_1w[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 AND below weekly EMA (downtrend) AND volume spike
            elif close[i] < s3_daily[i] and close[i] < ema_1w[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly S3 OR below weekly EMA (trend change)
            if close[i] < s3_daily[i] or close[i] < ema_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly R3 OR above weekly EMA (trend change)
            if close[i] > r3_daily[i] or close[i] > ema_1w[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals