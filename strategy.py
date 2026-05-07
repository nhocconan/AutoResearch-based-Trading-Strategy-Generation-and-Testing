#!/usr/bin/env python3
# 4h_1d_EMA_Crossover_Pullback
# Hypothesis: Use 1d EMA50 as primary trend filter, enter on 4h EMA21 pullbacks with volume confirmation.
# Works in bull markets (trend following) and bear markets (mean reversion off EMA50).
# Targets 20-40 trades/year on 4h timeframe with disciplined entries.
# Uses 4h primary timeframe and 1h for trend confirmation per experiment requirements.

name = "4h_1d_EMA_Crossover_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF as specified)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1h data for entry timing (lower timeframe precision)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 21:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h EMA21 for pullback entries
    ema_21_1h = pd.Series(close_1h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1h volume average for confirmation
    vol_avg_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_21_1h_4h = align_htf_to_ltf(prices, df_1h, ema_21_1h)
    vol_avg_1h_4h = align_htf_to_ltf(prices, df_1h, vol_avg_1h)
    
    # 4h volume spike filter
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_4h[i]) or np.isnan(ema_21_1h_4h[i]) or 
            np.isnan(vol_avg_1h_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Above 1d EMA50 trend, pullback to 1h EMA21 with volume
            if close[i] > ema_50_1d_4h[i] and close[i] <= ema_21_1h_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Below 1d EMA50 trend, bounce to 1h EMA21 with volume
            elif close[i] < ema_50_1d_4h[i] and close[i] >= ema_21_1h_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Break below 1h EMA21 or trend reversal
            if close[i] < ema_21_1h_4h[i] or close[i] < ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Break above 1h EMA21 or trend reversal
            if close[i] > ema_21_1h_4h[i] or close[i] > ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals