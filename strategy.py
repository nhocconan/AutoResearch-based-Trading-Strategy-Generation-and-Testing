#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla pivot breakout with 1w EMA trend filter and volume confirmation.
Long when price breaks above Camarilla R3 level AND 1w EMA50 > EMA200 AND volume > 2x 20-period average.
Short when price breaks below Camarilla S3 level AND 1w EMA50 < EMA200 AND volume > 2x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for longs, R3 for shorts).
Uses 1w HTF for trend strength (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
Camarilla pivots identify intraday support/resistance; EMA filter ensures we only trade with the weekly trend.
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
    
    # Calculate 1w EMA for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where we have enough data for Camarilla calculation (need previous day's OHLC)
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need previous day's OHLC for Camarilla calculation
        if i == 0:
            continue
            
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels for today based on yesterday's OHLC
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_r3 = prev_close + range_val * 1.1 / 4
        camarilla_s3 = prev_close - range_val * 1.1 / 4
        
        price = close[i]
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA50 > EMA200 AND volume spike
            if price > camarilla_r3 and ema_50_aligned[i] > ema_200_aligned[i] and volume[i] > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA50 < EMA200 AND volume spike
            elif price < camarilla_s3 and ema_50_aligned[i] < ema_200_aligned[i] and volume[i] > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < camarilla_s3:  # Long exit at Camarilla S3
                exit_signal = True
            elif position == -1 and price > camarilla_r3:  # Short exit at Camarilla R3
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R3S3_Breakout_1wEMA50_200_Trend_VolumeConfirmation_LevelExit"
timeframe = "1d"
leverage = 1.0