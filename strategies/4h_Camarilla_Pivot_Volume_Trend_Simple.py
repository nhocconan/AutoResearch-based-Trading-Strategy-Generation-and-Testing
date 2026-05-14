# [EXPERIMENT #153980] 4h_Camarilla_Pivot_Volume_Trend_Simple
# Hypothesis: Use Camarilla pivot levels from daily timeframe for support/resistance,
# enter on breakouts with volume confirmation, and filter by 4h trend direction.
# This combines price structure (pivots), momentum (volume), and trend (4h EMA)
# to work in both bull and bear markets by avoiding false breakouts.
# Target: 20-50 trades/year to stay under fee drag limits.

#!/usr/bin/env python3
name = "4h_Camarilla_Pivot_Volume_Trend_Simple"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = C + ((H-L) * 1.5/2), R3 = C + ((H-L) * 1.25/2), etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels (using formulas that work intraday)
    # R3 = Close + (Range * 1.1/2), S3 = Close - (Range * 1.1/2)
    # These are the most commonly used levels for breakouts
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 4h EMA for trend filter (slower to avoid whipsaws)
    close_series = pd.Series(close)
    ema_4h = close_series.ewm(span=50, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.8x 20-period average (higher threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above EMA (uptrend) AND volume spike
            if close[i] > r3_4h[i] and close[i] > ema_4h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below EMA (downtrend) AND volume spike
            elif close[i] < s3_4h[i] and close[i] < ema_4h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below EMA (trend change)
            if close[i] < s3_4h[i] or close[i] < ema_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above EMA (trend change)
            if close[i] > r3_4h[i] or close[i] > ema_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals