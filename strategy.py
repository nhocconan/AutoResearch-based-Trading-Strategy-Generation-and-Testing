#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily 50-period EMA trend with price position relative to EMA
# Price above EMA50 indicates bullish trend, below indicates bearish trend
# Enter long when price crosses above EMA50 with volume > 1.5x 20-period average
# Enter short when price crosses below EMA50 with volume > 1.5x 20-period average
# Exit when price crosses back below/above EMA50
# Works in both bull/bear markets: captures trends in either direction
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_EMA50_Trend_VolumeConfirmation_v1"
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
    
    # Calculate daily EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 calculation
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price crosses above EMA50 with volume confirmation
            if close[i] > ema_50_aligned[i] and close[i-1] <= ema_50_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below EMA50 with volume confirmation
            elif close[i] < ema_50_aligned[i] and close[i-1] >= ema_50_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below EMA50
            if close[i] < ema_50_aligned[i] and close[i-1] >= ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above EMA50
            if close[i] > ema_50_aligned[i] and close[i-1] <= ema_50_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals