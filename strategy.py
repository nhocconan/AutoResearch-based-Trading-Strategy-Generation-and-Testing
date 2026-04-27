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
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily EMA(10) for momentum filter (fast EMA for momentum)
    ema10_1d = pd.Series(df_1d['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev * 2) / 4
    range_ = high_prev - low_prev
    
    # Focus on R1/S1 for mean reversion, R3/S3 for breakout (balance of frequency and accuracy)
    r1 = pivot + range_ * 1.0833  # 1.0833 = (1.1 - 1) * 0.5 * 4
    s1 = pivot - range_ * 1.0833
    r3 = pivot + range_ * 1.25
    s3 = pivot - range_ * 1.25
    
    # Align levels to 4h timeframe
    ema34_aligned = ema34_1d_aligned
    ema10_aligned = ema10_1d_aligned
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.8 * 20-period average (balanced frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, pivots, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema10_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_aligned[i]
        ema_momentum = ema10_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Mean reversion at S1/R1: price touches level and reverses
            # Long: touch S1, close above it, with momentum alignment and volume spike
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and 
                ema_momentum > ema_trend and vol_spike_val):
                signals[i] = size
                position = 1
            # Short: touch R1, close below it, with momentum alignment and volume spike
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and 
                  ema_momentum < ema_trend and vol_spike_val):
                signals[i] = -size
                position = -1
            # Breakout at R3/S3: strong break with momentum and volume
            # Long: break above R3 with momentum and volume spike
            elif (high[i] > r3_aligned[i] and close[i] > r3_aligned[i] and 
                  ema_momentum > ema_trend and vol_spike_val):
                signals[i] = size
                position = 1
            # Short: break below S3 with momentum and volume spike
            elif (low[i] < s3_aligned[i] and close[i] < s3_aligned[i] and 
                  ema_momentum < ema_trend and vol_spike_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R1 (profit target) or momentum reverses
            if high[i] >= r1_aligned[i] or ema_momentum < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches S1 (profit target) or momentum reverses
            if low[i] <= s1_aligned[i] or ema_momentum > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1S1_R3S3_MeanRev_Breakout_1dEMA34_EMA10_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0