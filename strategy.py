#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 12h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S3 AND close < 12h EMA50 AND volume > 1.8x 20-period average.
Exit when price crosses Camarilla pivot point (PP).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Camarilla levels from 12h provide institutional pivot points proven on BTC/ETH.
12h EMA50 offers smooth trend filter for 6h timeframe alignment.
Volume confirmation at 1.8x ensures only significant breakouts are taken.
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
    
    # Load 12h data for Camarilla and EMA50 - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h timeframe
    # Using previous 12h bar's OHLC
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    # Set first values to NaN (no previous bar)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + range_val * 1.1 / 4.0
    S3 = pivot - range_val * 1.1 / 4.0
    PP = pivot  # Pivot point for exit
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    PP_aligned = align_htf_to_ltf(prices, df_12h, PP)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 12h EMA50 AND volume spike
            if (price > R3_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND close < 12h EMA50 AND volume spike
            elif (price < S3_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla pivot point (PP)
            if position == 1 and price < PP_aligned[i]:
                exit_signal = True
            elif position == -1 and price > PP_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0