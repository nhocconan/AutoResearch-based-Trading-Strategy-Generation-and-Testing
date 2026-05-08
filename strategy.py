#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Camarilla_R3S3_Breakout_12hTrend_1dVol"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data once for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla calculation (use previous completed 12h bar)
    prev_high_12h = np.roll(df_12h['high'].values, 1)
    prev_low_12h = np.roll(df_12h['low'].values, 1)
    prev_close_12h = np.roll(df_12h['close'].values, 1)
    prev_high_12h[0] = df_12h['high'].values[0]
    prev_low_12h[0] = df_12h['low'].values[0]
    prev_close_12h[0] = df_12h['close'].values[0]
    
    # Camarilla pivot levels calculation (R3, S3)
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    range_12h = prev_high_12h - prev_low_12h
    r3_12h = pivot_12h + (range_12h * 1.1 / 4)
    s3_12h = pivot_12h - (range_12h * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    vol_ma20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (vol_ma20_1d * 2.0)
    vol_spike_6h = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float)) > 0.5  # boolean to float
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(ema_50_6h[i]) or np.isnan(vol_spike_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and above 12h EMA50 (uptrend)
            long_cond = (close[i] > r3_6h[i] and vol_spike_6h[i] and close[i] > ema_50_6h[i])
            
            # Short entry: price breaks below S3 with volume spike and below 12h EMA50 (downtrend)
            short_cond = (close[i] < s3_6h[i] and vol_spike_6h[i] and close[i] < ema_50_6h[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal)
            if close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above R3 (reversal signal)
            if close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout on 6h with 12h EMA50 trend filter and 1d volume spike confirmation.
# Enters long when price breaks above R3 with 1d volume spike and price above 12h EMA50 (uptrend).
# Enters short when price breaks below S3 with 1d volume spike and price below 12h EMA50 (downtrend).
# Exits when price reverses back through S3/R3 respectively.
# Uses 12h for Camarilla calculation (structure) and trend, 1d for volume confirmation (fresh data).
# Discrete sizing (0.25) to minimize churn. Targets 15-30 trades/year on 6h timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).