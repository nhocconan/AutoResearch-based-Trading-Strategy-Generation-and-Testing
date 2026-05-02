#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# Uses Camarilla pivot levels from 4h for breakout signals, 4h EMA(50) for trend direction
# Volume spike (2.0x 20-period average) ensures participation and reduces false breakouts
# Only takes breakouts in the direction of the 4h trend to avoid counter-trend whipsaws
# Session filter (08-20 UTC) reduces noise trades
# Discrete position sizing 0.20 balances risk and minimizes fee churn
# Targets 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend
# 4h trend filter provides strong directional bias suitable for 1h timeframe

name = "1h_Camarilla_R3S3_4hTrend_VolumeSpike_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (R3, S3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot = (high_4h + low_4h + close_4h) / 3.0
    range_hl = high_4h - low_4h
    camarilla_r3 = pivot + (range_hl * 1.1 / 4.0)
    camarilla_s3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 1h volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla, EMA and volume MA)
    start_idx = 80  # max(20 for volume, 50 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 AND uptrend AND volume confirm
            if (close[i] > camarilla_r3_aligned[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND downtrend AND volume confirm
            elif (close[i] < camarilla_s3_aligned[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR trend reverses to downtrend
            if (close[i] < camarilla_s3_aligned[i] or 
                not uptrend):  # exited if price closes below 4h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR trend reverses to uptrend
            if (close[i] > camarilla_r3_aligned[i] or 
                not downtrend):  # exited if price closes above 4h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals