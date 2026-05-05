#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R3/S3) breakout with volume confirmation and ATR volatility filter
# Long when price breaks above 1d Camarilla R3 AND 12h HMA21 > previous 12h HMA21 (uptrend) AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d Camarilla S3 AND 12h HMA21 < previous 12h HMA21 (downtrend) AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Camarilla midpoint (R3/S3 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels from 1d provide daily structure that works in both bull and bear markets
# 12h HMA21 filter ensures we trade with the higher timeframe trend, reducing whipsaw
# Moderate volume confirmation (1.5x) validates breakout strength while avoiding excessive filtering

name = "12h_1dCamarillaR3S3_12hHMA21_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R3, S3, midpoint)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We use R3 and S3 as key levels
    high_low_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_s3 = close_1d - 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2.0
    
    # Align 1d Camarilla to 12h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Get 12h data ONCE before loop for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need at least 21 completed 12h bars for HMA21
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA21 (Hull Moving Average)
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = np.array([wma(close_12h[i:i+half_len], half_len) if i+half_len <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                           for i in range(len(raw_hma))])
    
    # Align 12h HMA21 to 12h timeframe (no additional delay needed for HMA)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3, 12h HMA21 rising (uptrend), volume confirmation, in session
            if (close[i] > camarilla_r3_aligned[i] and 
                hma_21_12h_aligned[i] > hma_21_12h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3, 12h HMA21 falling (downtrend), volume confirmation, in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  hma_21_12h_aligned[i] < hma_21_12h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals