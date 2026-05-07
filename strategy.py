#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla Pivot R3/S3 breakout with 1-day EMA34 trend filter and volume spike.
# Long when: Close > R3 (1d) AND Close > EMA34(1d) AND volume > 2.0 * EMA20(volume).
# Short when: Close < S3 (1d) AND Close < EMA34(1d) AND volume > 2.0 * EMA20(volume).
# Exit when price crosses back below/above EMA34(1d) or volume drops below EMA20(volume).
# Uses Camarilla levels from daily timeframe for institutional levels, EMA34 for trend filter.
# Volume spike requirement reduces false breakouts. Designed for low trade frequency (~25/year).
# Works in bull markets via upward breaks of R3 and in bear markets via downward breaks of S3.
name = "4h_Camarilla_R3S3_EMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA34 for exit
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Load 1d data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Using typical formula: R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low), etc.
    # But standard Camarilla uses: R3 = Close + 1.1*(High-Low), S3 = Close - 1.1*(High-Low)
    # We'll calculate for each day using previous day's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    
    # Avoid division by zero in first element
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    rang = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * rang
    s3 = prev_close_1d - 1.1 * rang
    
    # EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3 AND Close > EMA34(1d) AND volume spike
            long_condition = (close[i] > r3_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: Close < S3 AND Close < EMA34(1d) AND volume spike
            short_condition = (close[i] < s3_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA34(4h) OR Close < EMA34(1d) OR volume drops below EMA20
            if close[i] < ema_34[i] or close[i] < ema_34_1d_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA34(4h) OR Close > EMA34(1d) OR volume drops below EMA20
            if close[i] > ema_34[i] or close[i] > ema_34_1d_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals