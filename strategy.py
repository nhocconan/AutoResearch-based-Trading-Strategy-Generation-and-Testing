#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1w EMA50 Trend Filter and Volume Spike
# Long when price breaks above R3 (1w) AND price > 1w EMA50 (strong uptrend) AND volume spike
# Short when price breaks below S3 (1w) AND price < 1w EMA50 (strong downtrend) AND volume spike
# R3/S3 are strong Camarilla levels (PP ± range*1.1/2) for high-quality breaks with good risk/reward
# 1w EMA50 provides smoother trend filter than shorter EMAs, reducing whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 12h (primary timeframe as required)

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla levels and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1w bar (HLC of completed weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to use only completed weekly bar (look-ahead safety)
    high_1w_shifted = np.roll(high_1w, 1)
    low_1w_shifted = np.roll(low_1w, 1)
    close_1w_shifted = np.roll(close_1w, 1)
    
    # Calculate pivot point (PP) = (H+L+C)/3
    pp = (high_1w_shifted + low_1w_shifted + close_1w_shifted) / 3.0
    # Calculate range
    range_1w = high_1w_shifted - low_1w_shifted
    # Camarilla levels (R3/S3 = PP ± range*1.1/2)
    r3 = pp + (range_1w * 1.1 / 2.0)  # R3 = PP + range*1.1/2
    s3 = pp - (range_1w * 1.1 / 2.0)  # S3 = PP - range*1.1/2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation on 12h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND strong uptrend (price > 1w EMA50) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND strong downtrend (price < 1w EMA50) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 OR closes below 1w EMA50
            if close[i] < r3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S3 OR closes above 1w EMA50
            if close[i] > s3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals