#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R3/S3) breakout with 4h HMA21 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 4h HMA21 is rising AND 4h volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND 4h HMA21 is falling AND 4h volume > 2.0 * avg_volume(20)
# Exit when price returns to 1d Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels from higher timeframe (1d) provide strong intraday support/resistance
# HMA21 on 4h filters for trend direction while reducing lag
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "4h_1dCamarilla_R3S3_Breakout_4hHMA21_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R3, S3, PP)
    # Camarilla formulas: PP = (H+L+C)/3, Range = H-L
    # R3 = PP + Range * 1.1/4, S3 = PP - Range * 1.1/4
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pp_1d + (range_1d * 1.1 / 4.0)
    s3_1d = pp_1d - (range_1d * 1.1 / 4.0)
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 4h data ONCE before loop for HMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:  # Need at least 21 completed 4h bars for HMA21
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA21: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).rolling(window=half_n, min_periods=half_n).mean().values
    wma_full = pd.Series(close_4h).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_4h = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
    
    # Align 4h HMA21 to 4h timeframe (no additional delay needed for HMA)
    hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(hma_21_4h_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, HMA21 rising, volume spike
            if (close[i] > r3_1d_aligned[i] and close[i-1] <= r3_1d_aligned[i-1] and 
                hma_21_4h_aligned[i] > hma_21_4h_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, HMA21 falling, volume spike
            elif (close[i] < s3_1d_aligned[i] and close[i-1] >= s3_1d_aligned[i-1] and 
                  hma_21_4h_aligned[i] < hma_21_4h_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Camarilla pivot point (PP)
            if close[i] <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Camarilla pivot point (PP)
            if close[i] >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals