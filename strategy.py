#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla levels provide intraday support/resistance; breakout indicates momentum
# 4h EMA50 filters for higher timeframe trend alignment (works in bull/bear via trend following)
# Volume confirmation avoids false breakouts
# Session filter (08-20 UTC) reduces noise
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# Signal size: 0.20 discrete levels to minimize churn

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precompute once)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 1h Camarilla levels (R3, S3) from previous day
    # Need daily high/low/close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1h bar using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 formulas
    # R3 = close + 1.1*(high-low)/4
    # S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 1h (using previous day's values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 4h EMA50 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND below 4h EMA50 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 OR below 4h EMA50
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above R3 OR above 4h EMA50
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals