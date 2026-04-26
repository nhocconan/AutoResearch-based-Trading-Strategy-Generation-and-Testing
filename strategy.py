#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Camarilla pivot R1/S1 breakout on 1h with 4h EMA50 trend filter and volume confirmation (>1.5x 20-period MA).
Long when price breaks above R1 in 4h uptrend with volume spike. Short when price breaks below S1 in 4h downtrend with volume spike.
Uses discrete position sizing (0.20) to minimize fee churn. 
Camarilla levels derived from prior 1h OHLC. 
Designed to work in both bull and bear markets by following the 4h trend.
Target: 15-37 trades/year (60-150 total over 4 years).
Reduced trade frequency by tightening volume confirmation threshold from 1.3x to 1.5x and requiring EMA trend alignment for both entry and exit.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla calculation and 4h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    if len(df_1h) < 2 or len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1h OHLC
    # Using prior hour's close to avoid look-ahead
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # Shift by 1 to use prior hour's OHLC for current hour's levels
    close_1h_prev = np.roll(close_1h, 1)
    high_1h_prev = np.roll(high_1h, 1)
    low_1h_prev = np.roll(low_1h, 1)
    close_1h_prev[0] = np.nan
    high_1h_prev[0] = np.nan
    low_1h_prev[0] = np.nan
    
    # Camarilla R1, S1, R3, S3 levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_range = high_1h_prev - low_1h_prev
    r1 = close_1h_prev + camarilla_range * 1.1 / 12
    s1 = close_1h_prev - camarilla_range * 1.1 / 12
    r3 = close_1h_prev + camarilla_range * 1.1 / 4
    s3 = close_1h_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1h, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1h, s3)
    
    # 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA (tightened from 1.3x to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA + 20 for volume MA + 1 for Camarilla shift)
    start_idx = 71
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_4h[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h downtrend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_4h[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below R3 (strong reversal) OR 4h trend changes to downtrend
            if (close[i] < r3_aligned[i] or not uptrend_4h[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above S3 (strong reversal) OR 4h trend changes to uptrend
            if (close[i] > s3_aligned[i] or not downtrend_4h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0