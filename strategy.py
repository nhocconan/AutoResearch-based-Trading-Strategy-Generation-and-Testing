#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1h EMA50 trend filter
# Long when price breaks above Camarilla R3 with volume > 2.0x 20-period volume EMA AND price > 1h EMA50
# Short when price breaks below Camarilla S3 with volume > 2.0x 20-period volume EMA AND price < 1h EMA50
# Uses 1h EMA50 for trend alignment to reduce whipsaw vs pure breakout, targeting 20-50 trades/year on 4h.
# Volume spike filter (2.0x) is strict to avoid overtrading. Camarilla R3/S3 are strong breakout levels.
# Works in bull markets via longs in bullish 1h trend regime and bear markets via shorts in bearish 1h trend regime.

name = "4h_Camarilla_R3S3_1hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for EMA50 trend filter - ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    # Calculate 1h EMA50
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 1h EMA50 to 4h timeframe
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 from previous 1d bar
    camarilla_range = (high_1d - low_1d) * 1.1
    camarilla_r3 = close_1d + camarilla_range * 1.25 / 4  # R3 = close + 1.1*(high-low)*1.25/4
    camarilla_s3 = close_1d - camarilla_range * 1.25 / 4  # S3 = close - 1.1*(high-low)*1.25/4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2.0x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND price > 1h EMA50
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND price < 1h EMA50
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR price < 1h EMA50
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_50_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR price > 1h EMA50
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_50_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals