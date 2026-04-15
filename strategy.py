#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + 1d RSI + 1d Choppiness regime filter
# Uses 12h Kaufman Adaptive Moving Average to capture trend direction with low lag,
# 1d RSI to identify overbought/oversold conditions within the trend,
# and 1d Choppiness Index to avoid ranging markets. Works in both bull and bear
# by only taking trades in the direction of the 12h KAMA trend when the market
# is trending (Chop < 61.8) and RSI is not extreme.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Load 1d data for RSI and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10-period ER, 2-period fast, 30-period slow) on 12h
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    abs_change_12h = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])).reshape(-1, 1), axis=1)
    dir_12h = np.abs(close_12h - np.roll(close_12h, 10))
    vol_12h = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if hasattr(np.abs(np.diff(close_12h, prepend=close_12h[0])), 'shape') else np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])))
    # Simplified ER calculation for 1D array
    er_12h = dir_12h / (np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) + 1e-10) if len(close_12h.shape) > 1 else dir_12h / (np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0]))) + 1e-10)
    # Correct ER for 1D: Efficiency Ratio = |close - close[10]| / sum(|diff| over 10 periods)
    change_over_10 = np.abs(close_12h - np.roll(close_12h, 10))
    sum_abs_diff_10 = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i >= 10:
            sum_abs_diff_10[i] = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
        else:
            sum_abs_diff_10[i] = 1e-10
    er_12h = change_over_10 / (sum_abs_diff_10 + 1e-10)
    sc_12h = (er_12h * (2/2 - 30/30) + 30/30) ** 2  # sc = [ER*(fastest - slowest) + slowest]^2
    kama_12h = np.zeros_like(close_12h)
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    
    # Calculate RSI (14-period) on 1d
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Calculate Choppiness Index (14-period) on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Align all indicators to 12h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            continue
        
        # Long entry: price above KAMA (uptrend) + RSI not overbought + chop < 61.8 (trending)
        if (close[i] > kama_12h_aligned[i] and
            rsi_1d_aligned[i] < 70 and
            chop_aligned[i] < 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below KAMA (downtrend) + RSI not oversold + chop < 61.8 (trending)
        elif (close[i] < kama_12h_aligned[i] and
              rsi_1d_aligned[i] > 30 and
              chop_aligned[i] < 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or chop > 61.8 (ranging market)
        elif position == 1 and (close[i] < kama_12h_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_12h_aligned[i] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_KAMA_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0