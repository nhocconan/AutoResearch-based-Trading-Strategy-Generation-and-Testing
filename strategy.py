#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout + 1d EMA50 Trend + Volume Spike
# Long when price breaks above R3 with 1d uptrend and volume spike.
# Short when price breaks below S3 with 1d downtrend and volume spike.
# Camarilla levels provide precise support/resistance from prior day.
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades.
# Volume spike confirms institutional participation.
# Designed for 20-50 trades/year on 4h to minimize fee drag while capturing strong intraday moves.
# Works in bull markets via long breakouts and bear markets via short breakdowns.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: H' = (H-L)/1.1, then R4=close+1.5*H', R3=close+1.25*H', etc.
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    H_prime = (high_1d - low_1d) / 1.1
    R3 = close_1d_vals + 1.25 * H_prime
    S3 = close_1d_vals - 1.25 * H_prime
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 1d uptrend AND volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below S3 AND 1d downtrend AND volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 OR 1d trend turns down
            if (close[i] < R3_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above S3 OR 1d trend turns up
            if (close[i] > S3_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals