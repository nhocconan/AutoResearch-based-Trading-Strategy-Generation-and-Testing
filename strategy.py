#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (price > 4h EMA34 for long, < for short) and volume spike confirmation.
# Uses Camarilla R3/S3 levels (tighter than R4/S4) for higher probability entries. Trend filter ensures trades align with 4h momentum.
# Volume confirmation filters low-conviction breakouts. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# Discrete position sizing (0.20) to minimize fee churn.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeConfirm_v1"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from previous day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels (breakout continuation zones)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.05
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.05
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA34 and volume median
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 4h EMA34 direction
        uptrend = curr_close > ema_34_4h_aligned[i]
        downtrend = curr_close < ema_34_4h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Camarilla breakout conditions (R3/S3 for continuation)
        breakout_up = curr_close > camarilla_r3_aligned[i]   # break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i] # break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume confirmation AND in session
            if breakout_up and uptrend and volume_confirm and in_session:
                signals[i] = 0.20
                position = 1
            # Short: Breakout down AND downtrend AND volume confirmation AND in session
            elif breakout_down and downtrend and volume_confirm and in_session:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakout down (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout up (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals