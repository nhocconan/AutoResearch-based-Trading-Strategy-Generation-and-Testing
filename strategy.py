#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Fade_1dTrend_VolumeConfirmation
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance on 6h timeframe.
Fade (mean-revert) from R3/S3 when 1d trend confirms the reversal and volume spikes.
In bull markets: buy S3 fade, sell R3 fade. In bear markets: sell R3 fade, buy S3 fade.
Uses 6h for entry timing, 1d for trend filter and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation, trend filter, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: shift by 1 to use previous day's data
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + (range_1d * 1.1 / 4)
    camarilla_s3 = prev_close - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Trend filter: 1d EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (strong filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA + 20 for volume MA + 1 for shift)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Fade conditions: price touches/breaks R3 or S3 with volume
        touches_r3 = high[i] >= camarilla_r3_aligned[i]
        touches_s3 = low[i] <= camarilla_s3_aligned[i]
        
        if position == 0:
            # Long fade from S3: price touches S3, in uptrend, with volume
            if touches_s3 and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short fade from R3: price touches R3, in downtrend, with volume
            elif touches_r3 and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price moves back above S3 (mean reversion complete) or trend breaks
            if low[i] > camarilla_s3_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price moves back below R3 (mean reversion complete) or trend breaks
            if high[i] < camarilla_r3_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Fade_1dTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0