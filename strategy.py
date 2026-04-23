#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 level AND close > 4h EMA34 AND volume > 2.5x 24-period average.
Short when price breaks below Camarilla S3 level AND close < 4h EMA34 AND volume > 2.5x 24-period average.
Exit when price crosses Camarilla Pivot point (central level).
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-37 trades/year per symbol.
Camarilla R3/S3 levels (close ± 1.600 * daily range) provide stronger breakout validation.
4h EMA34 offers smooth trend filter with moderate lag. Volume confirmation at 2.5x ensures institutional-grade breakouts.
Designed to work in both bull and bear markets by using HTF trend filter and volatility-adjusted entries.
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
    
    # Load 4h data for EMA34 trend filter and Camarilla levels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 4h OHLC
    range_4h = high_4h - low_4h
    camarilla_r3_4h = close_4h + 1.600 * range_4h   # R3: close + 1.600 * range
    camarilla_s3_4h = close_4h - 1.600 * range_4h   # S3: close - 1.600 * range
    camarilla_pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    
    # Align HTF indicators to 1h timeframe
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
    
    # Volume average (24-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Ensure warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 4h EMA34 AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume[i] > 2.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND close < 4h EMA34 AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume[i] > 2.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla Pivot point
            if position == 1 and price < camarilla_pivot_aligned[i]:
                exit_signal = True
            elif position == -1 and price > camarilla_pivot_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_4hEMA34_VolumeSpike"
timeframe = "1h"
leverage = 1.0