#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2 (using prior 1d candle)
- Long: Close > R3 + volume > 2.0x 20-period avg + price > 1d EMA34
- Short: Close < S3 + volume > 2.0x 20-period avg + price < 1d EMA34
- Exit: Opposite breakout (Close < R3 for long, Close > S3 for short) or EMA34 trend flip
- Uses Camarilla for structure, volume for conviction, 1d EMA34 for HTF trend filter
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- EMA34 provides balanced trend filter to reduce false breakouts while maintaining responsiveness
"""

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
    
    # Volume confirmation: > 2.0x 20-period average (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate prior 1d Camarilla R3 and S3 levels
    # Need prior 1d OHLC for each 12h bar
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 12h timeframe (using prior 1d close for look-ahead safety)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r3_1d_aligned[i]) or
            np.isnan(camarilla_s3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > R3 + volume confirmation + price > 1d EMA34
            if (close[i] > camarilla_r3_1d_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 + volume confirmation + price < 1d EMA34
            elif (close[i] < camarilla_s3_1d_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < R3 OR price < 1d EMA34 (trend flip)
            if close[i] < camarilla_r3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > S3 OR price > 1d EMA34 (trend flip)
            if close[i] > camarilla_s3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0