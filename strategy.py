#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA(50) trend filter and volume confirmation
# Uses tighter Camarilla levels (R3/S3) for more precise entries, 12h EMA(50) for stronger trend filter
# Volume spike (1.8x 24-period average) ensures participation while reducing false signals
# Only takes breakouts in the direction of the 12h trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# 12h trend filter provides medium-term bias suitable for 4h timeframe, working in both bull and bear markets

name = "4h_Camarilla_R3S3_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla pivots and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (R3, S3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_hl = high_12h - low_12h
    camarilla_r3 = pivot + (range_hl * 1.1 / 4.0)  # R3 level
    camarilla_s3 = pivot - (range_hl * 1.1 / 4.0)  # S3 level
    
    # Calculate 12h EMA(50) for trend filter
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 4h volume confirmation (1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla, EMA and volume MA)
    start_idx = 80  # max(24 for volume, 50 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 AND uptrend AND volume confirm
            if (close[i] > camarilla_r3_aligned[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND downtrend AND volume confirm
            elif (close[i] < camarilla_s3_aligned[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR trend reverses to downtrend
            if (close[i] < camarilla_s3_aligned[i] or 
                not uptrend):  # exited if price closes below 12h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR trend reverses to uptrend
            if (close[i] > camarilla_r3_aligned[i] or 
                not downtrend):  # exited if price closes above 12h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals