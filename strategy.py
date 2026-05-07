#!/usr/bin/env python3
name = "1h_Stochastic_Trend_With_Volume"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h Stochastic (14,3,3) for momentum
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # 1h volume filter: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for volume MA and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(d_percent[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above 4h EMA50, Stochastic rising from oversold, volume
            if (close[i] > ema50_4h_aligned[i] and 
                k_percent[i] < 30 and d_percent[i] < 30 and 
                k_percent[i] > k_percent[i-1] and d_percent[i] > d_percent[i-1] and
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price below 4h EMA50, Stochastic falling from overbought, volume
            elif (close[i] < ema50_4h_aligned[i] and 
                  k_percent[i] > 70 and d_percent[i] > 70 and 
                  k_percent[i] < k_percent[i-1] and d_percent[i] < d_percent[i-1] and
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price below 4h EMA50 or Stochastic overbought
            if close[i] < ema50_4h_aligned[i] or d_percent[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price above 4h EMA50 or Stochastic oversold
            if close[i] > ema50_4h_aligned[i] or d_percent[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h trend following with 4h EMA50 trend filter and Stochastic(14,3,3) momentum.
# Long when price > 4h EMA50, Stochastic rising from oversold (<30), and volume confirms.
# Short when price < 4h EMA50, Stochastic falling from overbought (>70), and volume confirms.
# Uses 4h timeframe for trend to avoid whipsaws, 1h for entry timing with Stochastic.
# Volume filter (>1.3x average) ensures conviction. Session filter (8-20 UTC) reduces noise.
# Target: 15-35 trades/year to minimize fee drag while capturing sustained moves in both bull and bear markets.