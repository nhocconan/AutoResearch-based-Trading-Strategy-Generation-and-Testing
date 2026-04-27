# 12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Weekly trend (1w EMA50) filters Camarilla R3/S3 breakouts on 12h timeframe, with volume confirmation to avoid false breakouts. Works in bull/bear by only taking trades aligned with weekly trend, reducing whipsaw.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous week's close for EMA50 trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Previous day's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels (R3 and S3)
    range_hl = prev_high - prev_low
    R3 = prev_close + range_hl * 1.1 / 4
    S3 = prev_close - range_hl * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0 * 24-period average (24*12h = 12 days lookback)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 24) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 + weekly uptrend + volume spike
            if (close[i] > R3_12h[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below S3 + weekly downtrend + volume spike
            elif (close[i] < S3_12h[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S3 (reversal) or trend changes to down
            if (close[i] < S3_12h[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R3 (reversal) or trend changes to up
            if (close[i] > R3_12h[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0