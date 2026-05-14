#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h HMA21 trend filter and volume confirmation (>1.8x average)
# Uses 12h HTF for HMA21 to capture intermediate trend and reduce false breakouts in choppy markets.
# Camarilla R3/S3 from 4h provides proven intraday reversal/continuation levels with good historical performance.
# Volume confirmation at 1.8x average ensures strong participation while limiting trades (~30-60/year).
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

name = "4h_Camarilla_R3_S3_Breakout_12hHMA21_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels R3 and S3 from 4h timeframe (using prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Prior 4h bar's high, low, close for Camarilla calculation
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Camarilla R3 and S3 levels (proven breakout/continuation levels)
    camarilla_r3_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4
    camarilla_s3_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (they are already 4h, but align for safety)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # 12h HMA21 for trend filter (intermediate trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Hull Moving Average calculation
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Volume confirmation: 1.8x 20-period average (balanced threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND price > 12h HMA21 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > hma_21_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND price < 12h HMA21 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < hma_21_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below S3 OR price < 12h HMA21
            if close[i] < camarilla_s3_aligned[i] or close[i] < hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above R3 OR price > 12h HMA21
            if close[i] > camarilla_r3_aligned[i] or close[i] > hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals