#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend filter
    ema_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and 4h uptrend
            if (close[i] > r3_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5 and 
                close[i] > ema_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume spike and 4h downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5 and 
                  close[i] < ema_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price falls back below R3 or trend changes
            if (close[i] < r3_aligned[i] or 
                close[i] < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price rises back above S3 or trend changes
            if (close[i] > s3_aligned[i] or 
                close[i] > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(34) trend filter and volume confirmation.
# Camarilla R3 and S3 represent strong intraday support/resistance levels.
# Breakouts above R3 with volume indicate bullish momentum; breakdowns below S3 indicate bearish momentum.
# The 4h EMA(34) filter ensures we only trade in the direction of the higher timeframe trend,
# reducing whipsaws during sideways markets. Volume confirmation adds validity to breakouts.
# Session filter (08-20 UTC) avoids low-volume Asian session noise.
# Position size 0.20 balances risk and return while keeping trade frequency ~20-40/year.