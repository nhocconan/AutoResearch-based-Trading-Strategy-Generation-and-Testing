#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Camarilla levels identify key intraday support/resistance; breakouts at R3/S3 with trend alignment capture momentum
# Works in bull/bear: 12h EMA50 ensures we trade with higher timeframe trend to avoid whipsaws
# Volume spike (>1.5x 20-period EMA) confirms breakout authenticity
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
# Using discrete position sizing (0.25) to reduce fee churn

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 1d bar
    # R3 = close + 1.1*(high-low)*1.25/2, S3 = close - 1.1*(high-low)*1.25/2
    # Using prior 1d bar to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = 1.1 * (high_1d - low_1d) * 1.25 / 2
    r3 = close_1d + camarilla_range
    s3 = close_1d - camarilla_range
    
    # Align Camarilla levels to 6h timeframe (using prior 1d bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Camarilla R3/S3 breakout signals with 12h trend filter
        # Long: price breaks above R3 + price above 12h EMA50 + volume spike
        # Short: price breaks below S3 + price below 12h EMA50 + volume spike
        if position == 0:
            if (close[i] > r3_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < s3_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below R3 (failed breakout) OR price below 12h EMA50
            if close[i] < r3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above S3 (failed breakdown) OR price above 12h EMA50
            if close[i] > s3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals