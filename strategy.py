#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 12h EMA trend filter and volume spike.
# Uses Camarilla R3/S3 levels for mean-reversion entries in ranging markets,
# 12h EMA50 for trend filter (only trade with trend), and volume > 1.5x average for confirmation.
# Works in both bull and bear by only taking trades in the direction of the 12h trend.
# Target: 20-50 trades per year to minimize fee drag.

name = "4h_Camarilla_R3S3_Reversal_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from previous day
    # Using prior day's OHLC (approximated as prior candle in 4h timeframe)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    
    # Camarilla calculations
    R3 = close + (high - low) * 1.1 / 4
    S3 = close - (high - low) * 1.1 / 4
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = ema_50_12h[1:] > ema_50_12h[:-1]
    trend_12h_up = np.concatenate([[False], trend_12h_up])
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade with 12h trend
            if trend_12h_up_aligned[i]:
                # Uptrend: look for reversals at S3 (support)
                if close[i] <= S3[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            else:
                # Downtrend: look for reversals at R3 (resistance)
                if close[i] >= R3[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reaches midpoint or trend changes
            midpoint = (close[i-1] + S3[i]) / 2  # approximate midpoint
            if close[i] >= midpoint or not trend_12h_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches midpoint or trend changes
            midpoint = (close[i-1] + R3[i]) / 2  # approximate midpoint
            if close[i] <= midpoint or trend_12h_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals