#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter
Hypothesis: Camarilla R3/S3 levels from 1d act as strong intraday support/resistance. 
Break above R3 with 1w uptrend and volume spike = long. Break below S3 with 1w downtrend 
and volume spike = short. Weekly trend filter avoids counter-trend trades in bear markets.
Target: 7-25 trades/year per symbol (30-100 total over 4 years).
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    uptrend = close > ema_34_1w_aligned
    downtrend = close < ema_34_1w_aligned
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels: based on previous day's range
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align daily Camarilla levels to 1d timeframe (no further alignment needed)
    camarilla_r3_aligned = camarilla_r3  # Already at 1d frequency
    camarilla_s3_aligned = camarilla_s3  # Already at 1d frequency
    
    # Volume confirmation: volume > 2.0x 20-period MA (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_confirm[i]) or 
            i >= len(camarilla_r3_aligned) or i >= len(camarilla_s3_aligned) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, with 1w uptrend and volume confirmation
            if close[i] > camarilla_r3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, with 1w downtrend and volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Camarilla S3 (reversal) OR 1w trend changes to downtrend
            if close[i] < camarilla_s3_aligned[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Camarilla R3 (reversal) OR 1w trend changes to uptrend
            if close[i] > camarilla_r3_aligned[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0