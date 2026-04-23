#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 Breakout with 1w EMA50 Trend Filter and Volume Spike
- Camarilla R3/S3 levels from 1w act as stronger weekly support/resistance; breakout with volume indicates strong continuation
- 1w EMA50 defines the long-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.8x 20-period average) reduces false breakouts
- Designed for 1d timeframe to capture medium-term breakouts with low frequency (target: 15-25 trades/year)
- Uses tighter volume confirmation (1.8x) and weekly Camarilla levels (R3/S3) to reduce trade frequency vs daily versions
- Works in bull via long breakouts above R3 and in bear via short breakdowns below S3
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
    
    # Calculate 1w Camarilla pivots (based on previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R3/S3 = C ± (H-L)*1.1/4 (inner levels)
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align to 1d timeframe (use previous week's levels for breakout)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.8x 20-period average (~1 month)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # need 1w pivots, 1w EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 1w EMA50 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below 1w EMA50 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Camarilla H4/L4 levels (extreme levels) OR crosses 1w EMA50
            exit_signal = False
            # Calculate H4/L4 for exit (extreme levels)
            camarilla_h4 = close_1w + (high_1w - low_1w) * 1.1 / 2
            camarilla_l4 = close_1w - (high_1w - low_1w) * 1.1 / 2
            camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
            camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
            
            if position == 1:
                # Exit long when price > H4 OR < 1w EMA50
                if close[i] > camarilla_h4_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price < L4 OR > 1w EMA50
                if close[i] < camarilla_l4_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0