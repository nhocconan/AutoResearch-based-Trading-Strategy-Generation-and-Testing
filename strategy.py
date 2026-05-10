# 12h_Volume_Imbalance_Reversal
# Hypothesis: Volume imbalance at extreme price levels signals exhaustion and reversal.
# Long when price makes new low but volume declines (seller exhaustion).
# Short when price makes new high but volume declines (buyer exhaustion).
# Uses 1d ATR for dynamic thresholds and 1w trend filter to avoid counter-trend trades.
# Targets 20-40 trades/year (80-160 total) to minimize fee drag.

name = "12h_Volume_Imbalance_Reversal"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume imbalance detection
    # Compare current volume to average volume of last 20 periods
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price extreme detection using ATR-based bands
    # Upper band: highest high of last 10 periods + 0.5 * ATR
    # Lower band: lowest low of last 10 periods - 0.5 * ATR
    highest_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    upper_band = highest_high + 0.5 * atr_14_aligned
    lower_band = lowest_low - 0.5 * atr_14_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ATR(14), EMA50, and volume average
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade in direction of 1w trend
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        # Volume condition: current volume < 70% of average (indicating exhaustion)
        vol_exhaustion = volume[i] < vol_avg[i] * 0.7
        
        if position == 0:
            # Long: price at or below lower band + volume exhaustion + uptrend filter
            if low[i] <= lower_band[i] and vol_exhaustion and uptrend_1w:
                signals[i] = 0.25
                position = 1
            # Short: price at or above upper band + volume exhaustion + downtrend filter
            elif high[i] >= upper_band[i] and vol_exhaustion and downtrend_1w:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above midpoint or volume returns
            midpoint = (upper_band[i] + lower_band[i]) / 2
            vol_recovery = volume[i] > vol_avg[i] * 0.9
            if close[i] > midpoint or vol_recovery:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below midpoint or volume returns
            midpoint = (upper_band[i] + lower_band[i]) / 2
            vol_recovery = volume[i] > vol_avg[i] * 0.9
            if close[i] < midpoint or vol_recovery:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals