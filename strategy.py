#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike.
# Uses Camarilla pivot levels from prior 12h bar: long when price breaks above R3 with 12h uptrend and volume spike,
# short when breaks below S3 with 12h downtrend and volume spike. Volume filter: current volume > 2x 20-period average.
# Designed for 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 12h trend and requiring volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar: R3, S3
    # R3 = close + 1.1*(high - low), S3 = close - 1.1*(high - low)
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h)
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h)
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar to close)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above R3 AND 12h uptrend AND volume spike
        if (close[i] > r3_12h_aligned[i] and 
            close[i] > ema50_12h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below S3 AND 12h downtrend AND volume spike
        elif (close[i] < s3_12h_aligned[i] and 
              close[i] < ema50_12h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0