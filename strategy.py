#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 Breakout with 1d EMA50 Trend Filter and Volume Spike
- Camarilla R3/S3 levels from 1d act as stronger support/resistance than R1/S1; breakout with volume indicates strong continuation
- 1d EMA50 defines the long-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 2.0x 24-period MA) reduces false breakouts
- Designed for 12h timeframe to capture medium-term breakouts with controlled frequency (target: 12-37 trades/year)
- Uses tighter volume confirmation (2.0x) and inner Camarilla levels (R3/S3) to reduce trade frequency vs previous versions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 = C ± (H-L)*1.1/4 (inner levels)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 12h timeframe (use previous day's levels for breakout)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 24)  # need 1d pivots, 1d EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 1d EMA50 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below 1d EMA50 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Camarilla H4/L4 levels (extreme levels) OR crosses 1d EMA50
            exit_signal = False
            # Calculate H4/L4 for exit (extreme levels)
            camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
            camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
            camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            
            if position == 1:
                # Exit long when price > H4 OR < 1d EMA50
                if close[i] > camarilla_h4_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price < L4 OR > 1d EMA50
                if close[i] < camarilla_l4_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0