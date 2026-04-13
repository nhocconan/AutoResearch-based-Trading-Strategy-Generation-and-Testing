#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w Bollinger squeeze + RSI mean reversion.
# Long: 1w Bollinger Band Width < 20th percentile (squeeze) + RSI(14) < 30 on 1d.
# Short: 1w Bollinger Band Width < 20th percentile (squeeze) + RSI(14) > 70 on 1d.
# Exit: RSI crosses back above 50 (long) or below 50 (short).
# Uses volatility contraction (squeeze) to anticipate expansion in mean-reverting markets.
# Works in both bull/bear as mean reversion persists across regimes.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 1w data for Bollinger Band Width (volatility squeeze detection)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Bollinger Bands (20, 2) on weekly
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = np.full(len(close_1w), np.nan)
    for i in range(bb_length, len(close_1w)):
        basis[i] = np.mean(close_1w[i-bb_length:i])
    
    # Deviation (standard deviation)
    dev = np.full(len(close_1w), np.nan)
    for i in range(bb_length, len(close_1w)):
        dev[i] = np.std(close_1w[i-bb_length:i])
    
    # Upper and lower bands
    upper = basis + (dev * bb_mult)
    lower = basis - (dev * bb_mult)
    
    # Bollinger Band Width (normalized)
    bbwidth = np.full(len(close_1w), np.nan)
    for i in range(bb_length, len(close_1w)):
        if basis[i] != 0:
            bbwidth[i] = (upper[i] - lower[i]) / basis[i]
    
    # Percentile rank of BBWidth (20-period lookback)
    bbwidth_percentile = np.full(len(close_1w), np.nan)
    lookback = 20
    for i in range(lookback, len(close_1w)):
        window = bbwidth[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bbwidth_percentile[i] = (np.sum(valid <= bbwidth[i]) / len(valid)) * 100
    
    # Squeeze condition: BBWidth < 20th percentile
    squeeze = np.full(len(close_1w), False)
    for i in range(lookback, len(close_1w)):
        if not np.isnan(bbwidth_percentile[i]):
            squeeze[i] = bbwidth_percentile[i] < 20.0
    
    # 1d RSI calculation
    rsi_length = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Smoothed average gain/loss (Wilder's smoothing)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_length, n):
        if i == rsi_length:
            avg_gain[i] = np.mean(gain[i-rsi_length+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_length+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_length-1) + gain[i]) / rsi_length
            avg_loss[i] = (avg_loss[i-1] * (rsi_length-1) + loss[i]) / rsi_length
    
    # RSI calculation
    rsi = np.full(n, np.nan)
    for i in range(rsi_length, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100.0 if avg_gain[i] > 0 else 0.0
    
    # Align 1w squeeze signal to daily
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # Warmup period
        # Skip if any required data is not ready
        if np.isnan(squeeze_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        is_squeeze = squeeze_aligned[i] > 0.5  # Boolean from float alignment
        
        if position == 0:
            # Long: squeeze + RSI oversold (< 30)
            if is_squeeze and (rsi[i] < 30):
                position = 1
                signals[i] = position_size
            # Short: squeeze + RSI overbought (> 70)
            elif is_squeeze and (rsi[i] > 70):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses back above 50 (mean reversion complete)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses back below 50 (mean reversion complete)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Bollinger_Squeeze_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0