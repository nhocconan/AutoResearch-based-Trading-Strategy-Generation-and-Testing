#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation
# Uses Camarilla pivot levels from 12h for structure, 12h EMA(34) for trend direction
# Volume spike (2x 20-period average) ensures participation and reduces false breakouts
# Only takes breakouts in the direction of the 12h trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend

name = "6h_Camarilla_R3S3_12hTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla pivot, EMA trend, and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (R3, S3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3 = pivot + range_12h * 1.1 / 2.0
    s3 = pivot - range_12h * 1.1 / 2.0
    
    # Calculate 12h EMA(34) for trend filter
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 6h
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Calculate 6h volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA and volume MA)
    start_idx = 54  # max(34 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND uptrend AND volume confirm
            if (close[i] > r3_aligned[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND downtrend AND volume confirm
            elif (close[i] < s3_aligned[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 OR trend reverses to downtrend
            if (close[i] < s3_aligned[i] or 
                not uptrend):  # exited if price closes below 12h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend reverses to uptrend
            if (close[i] > r3_aligned[i] or 
                not downtrend):  # exited if price closes above 12h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals