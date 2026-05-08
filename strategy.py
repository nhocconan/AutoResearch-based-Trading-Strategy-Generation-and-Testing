#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily KAMA for trend and Bollinger Bands for mean-reversion signals.
# Long when price touches lower Bollinger Band (20,2) and KAMA is rising (uptrend).
# Short when price touches upper Bollinger Band (20,2) and KAMA is falling (downtrend).
# Uses daily KAMA to filter trend direction and reduce whipsaw. Bollinger Bands provide
# mean-reversion entries in ranging markets, while KAMA filters out counter-trend moves.
# Designed for low trade frequency (20-40/year) to minimize fee drag and capture mean-reversion
# within the dominant trend.

name = "4h_KAMA_Bollinger_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend (KAMA) and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d))
    change = np.concatenate([[0], change])  # align with close_1d index
    volatility = np.abs(np.diff(close_1d, 1))
    volatility = np.concatenate([[0], volatility])
    
    # Sum of absolute changes over 10 periods for ER numerator
    change_sum = np.convolve(change, np.ones(10), mode='full')
    change_sum = change_sum[9:len(change_sum)-0]  # valid part
    change_sum = np.concatenate([np.zeros(9), change_sum])  # align
    
    # Sum of absolute changes over 10 periods for ER denominator
    volatility_sum = np.convolve(volatility, np.ones(10), mode='full')
    volatility_sum = volatility_sum[9:len(volatility_sum)-0]
    volatility_sum = np.concatenate([np.zeros(9), volatility_sum])
    
    # Avoid division by zero
    er = np.where(volatility_sum != 0, change_sum / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Daily Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = np.convolve(close_1d, np.ones(bb_period)/bb_period, mode='same')
    # Handle edges
    sma[:bb_period-1] = np.nan
    sma[-bb_period+1:] = np.nan
    
    # Calculate rolling std using convolution for efficiency
    def rolling_std(arr, window):
        # Using method: sqrt(E[X^2] - E[X]^2)
        arr_sq = arr ** 2
        mean = np.convolve(arr, np.ones(window)/window, mode='same')
        mean_sq = np.convolve(arr_sq, np.ones(window)/window, mode='same')
        var = mean_sq - mean ** 2
        # Handle edges
        var[:window-1] = np.nan
        var[-window+1:] = np.nan
        return np.sqrt(np.maximum(var, 0))
    
    std = rolling_std(close_1d, bb_period)
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    
    # Align daily indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma)  # for exit
    
    # Bollinger Band width for regime filter (optional, can add later)
    # bb_width = (upper_bb - lower_bb) / sma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for calculations
    start_idx = max(30, bb_period)  # ensure BB and KAMA ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(sma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower BB and KAMA is rising (uptrend)
            if (low[i] <= lower_bb_aligned[i] and  # touch or penetrate lower band
                kama_aligned[i] > kama_aligned[i-1]):  # KAMA rising
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB and KAMA is falling (downtrend)
            elif (high[i] >= upper_bb_aligned[i] and  # touch or penetrate upper band
                  kama_aligned[i] < kama_aligned[i-1]):  # KAMA falling
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to SMA (mean reversion complete) or KAMA turns down
            if (high[i] >= sma_aligned[i] or  # price back to mean
                kama_aligned[i] < kama_aligned[i-1]):  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to SMA or KAMA turns up
            if (low[i] <= sma_aligned[i] or  # price back to mean
                kama_aligned[i] > kama_aligned[i-1]):  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals