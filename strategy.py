#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike + 1d EMA50 trend filter
# TRIX (Triple Exponential Average) filters noise and identifies momentum shifts.
# Combined with volume spike and daily trend, it avoids whipsaws in both bull and bear markets.
# Low trade frequency due to strict TRIX crossover + volume confirmation requirement.
name = "4h_TRIX_1dEMA50_Trend_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX on close (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # Volume filter: current volume > 2.0x 20-period average volume (strict to reduce trades)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 15  # Need enough data for TRIX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(trix[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX crossover signals
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        trend_up = close[i] > ema_50_4h[i]
        trend_down = close[i] < ema_50_4h[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + uptrend + volume confirmation
            if trix_cross_up and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + downtrend + volume confirmation
            elif trix_cross_down and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or trend reversal
            if trix_cross_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or trend reversal
            if trix_cross_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals