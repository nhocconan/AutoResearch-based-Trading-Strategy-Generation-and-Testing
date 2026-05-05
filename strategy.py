#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R3 level AND 12h EMA50 > 12h EMA200 (uptrend) AND volume > 1.5x 20-period average
# Short when price breaks below 4h Camarilla S3 level AND 12h EMA50 < 12h EMA200 (downtrend) AND volume > 1.5x 20-period average
# Exit when price crosses 4h Camarilla pivot point (mean reversion)
# Uses 4h primary timeframe with 12h HTF for trend filter and 1d HTF for volume mean (more stable)
# Volume confirmation ensures breakouts have conviction
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 and EMA200 for trend filter
    def calculate_ema(data, span):
        return pd.Series(data).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_50_12h = calculate_ema(close_12h, 50)
    ema_200_12h = calculate_ema(close_12h, 200)
    
    # Align EMAs to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Get 1d data ONCE before loop for volume mean (more stable than 4h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume 20-period mean
    if len(volume_1d) >= 20:
        vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    else:
        vol_ma_20_1d_aligned = np.full(n, np.nan)
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels
    camarilla_r3_4h = close_4h + (1.1 * (high_4h - low_4h) / 2)
    camarilla_s3_4h = close_4h - (1.1 * (high_4h - low_4h) / 2)
    camarilla_pivot_4h = (high_4h + low_4h + close_4h) / 3  # Standard pivot point
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
    
    # Volume confirmation: volume > 1.5x 1d volume 20-period mean
    if len(volume) >= 20 and not np.all(np.isnan(vol_ma_20_1d_aligned)):
        volume_filter = volume > (1.5 * vol_ma_20_1d_aligned)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(camarilla_r3_4h_aligned[i]) or 
            np.isnan(camarilla_s3_4h_aligned[i]) or 
            np.isnan(camarilla_pivot_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND uptrend (EMA50 > EMA200) AND volume spike
            if (close[i] > camarilla_r3_4h_aligned[i] and 
                ema_50_12h_aligned[i] > ema_200_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND downtrend (EMA50 < EMA200) AND volume spike
            elif (close[i] < camarilla_s3_4h_aligned[i] and 
                  ema_50_12h_aligned[i] < ema_200_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals