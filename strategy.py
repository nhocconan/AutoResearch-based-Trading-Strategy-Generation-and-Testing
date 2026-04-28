#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla pivot levels with volume confirmation and 1w HMA trend filter.
# Enter long when price breaks above weekly R3 with volume spike and above 1w HMA21.
# Enter short when price breaks below weekly S3 with volume spike and below 1w HMA21.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 total trades over 4 years.

name = "1d_Camarilla_R3S3_1wHMA21_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    R3 = np.full(n_1w, np.nan)
    S3 = np.full(n_1w, np.nan)
    PP = np.full(n_1w, np.nan)
    
    for i in range(n_1w):
        # Camarilla pivot calculation
        PP[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        range_1w = high_1w[i] - low_1w[i]
        R3[i] = PP[i] + range_1w * 1.1 / 4.0
        S3[i] = PP[i] - range_1w * 1.1 / 4.0
    
    # Forward fill to get most recent pivot levels
    R3 = pd.Series(R3).ffill().values
    S3 = pd.Series(S3).ffill().values
    PP = pd.Series(PP).ffill().values
    
    # Align 1w Camarilla levels to 1d timeframe with 1-bar delay for confirmation
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    
    # Calculate 1w HMA21 for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_1w_series = pd.Series(close_1w)
    wma_half = close_1w_series.rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(1, half_len+1)) / np.arange(1, half_len+1).sum(), raw=False
    ).values
    wma_full = close_1w_series.rolling(window=21, min_periods=21).apply(
        lambda x: np.dot(x, np.arange(1, 22)) / np.arange(1, 22).sum(), raw=False
    ).values
    
    # 2 * WMA(10.5) - WMA(21)
    raw_hma = 2 * wma_half - wma_full
    # WMA(sqrt(21)) of the above
    hma_21 = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=False
    ).values
    
    # Align HMA to 1d timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w HMA21
        above_hma = close[i] > hma_21_aligned[i]
        below_hma = close[i] < hma_21_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite pivot level or trend reversal
        long_exit = close[i] < S3_aligned[i] or below_hma
        short_exit = close[i] > R3_aligned[i] or above_hma
        
        # Handle entries and exits
        if long_breakout and above_hma and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and below_hma and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals