#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    R3 = np.zeros(len(close_1d))
    S3 = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        R3[i] = prev_close + range_val * 1.1 / 4
        S3[i] = prev_close - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: price breaks above R3 AND weekly uptrend (price > EMA50) AND volume confirmation
            if (close[i] > R3_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND weekly downtrend (price < EMA50) AND volume confirmation
            elif (close[i] < S3_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below S3 (reversal) OR weekly trend turns down
            if (close[i] < S3_aligned[i]) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above R3 (reversal) OR weekly trend turns up
            if (close[i] > R3_aligned[i]) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as strong support/resistance. 
# Breakouts with volume confirmation and weekly trend filter capture institutional moves.
# Long when price > R3, weekly uptrend, and volume spike. 
# Short when price < S3, weekly downtrend, and volume spike.
# Exits when price reverses to opposite level or weekly trend changes.
# Weekly trend filter reduces whipsaws in sideways markets. 
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.