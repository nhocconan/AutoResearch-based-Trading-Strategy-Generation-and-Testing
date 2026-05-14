#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm
Hypothesis: Camarilla R3/S3 breakout on 6h with volume confirmation and 12h EMA50 trend filter.
R3/S3 represent stronger reversal/breakout levels than R1/S1. Volume confirmation ensures
institutional participation. 12h EMA50 filter avoids counter-trend trades. Uses discrete sizing (0.25)
to minimize fee drag. Target: 50-150 trades over 4 years. Works in bull via breakout continuation
and in bear via mean-reversion fading at extreme levels (though primary logic is breakout).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 6h (based on previous bar's range)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1 / 4
    s3 = prev_close - range_hl * 1.1 / 4
    r4 = prev_close + range_hl * 1.1 / 2
    s4 = prev_close - range_hl * 1.1 / 2
    
    # Volume confirmation: volume > 1.5x 20-period median (more robust than mean)
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 1.5)
    
    # Load 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for volume median, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above R3 with volume confirmation and 12h uptrend
        long_condition = (close[i] > r3[i]) and volume_confirm[i] and (close[i] > ema_50_12h_aligned[i])
        # Short logic: break below S3 with volume confirmation and 12h downtrend
        short_condition = (close[i] < s3[i]) and volume_confirm[i] and (close[i] < ema_50_12h_aligned[i])
        
        # Exit logic: opposite Camarilla level (S3/R3) or trend reversal
        exit_long = (close[i] < s3[i]) or (close[i] < ema_50_12h_aligned[i])
        exit_short = (close[i] > r3[i]) or (close[i] > ema_50_12h_aligned[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0