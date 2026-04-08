#!/usr/bin/env python3
# 1d_ema_crossover_1w_trend_volume
# Hypothesis: Use EMA crossovers on daily timeframe for entries, filtered by weekly EMA trend.
# EMA crossovers capture momentum shifts, while weekly trend filter ensures we trade with the higher timeframe trend.
# Volume confirmation adds confirmation to reduce false signals.
# Target: 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_crossover_1w_trend_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate EMA crossovers on daily timeframe - 20 and 50 period
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average for confirmation - 20 period
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA crossover reverses OR trend turns against us
            if (ema_20[i] < ema_50[i]) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA crossover reverses OR trend turns against us
            if (ema_20[i] > ema_50[i]) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA 20 crosses above EMA 50 with uptrend and volume confirmation
            if (ema_20[i] > ema_50[i]) and (ema_20[i-1] <= ema_50[i-1]) and \
               (close[i] > ema_50_1w_aligned[i]) and (volume[i] > vol_ma[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: EMA 20 crosses below EMA 50 with downtrend and volume confirmation
            elif (ema_20[i] < ema_50[i]) and (ema_20[i-1] >= ema_50[i-1]) and \
                 (close[i] < ema_50_1w_aligned[i]) and (volume[i] > vol_ma[i]):
                position = -1
                signals[i] = -0.25
    
    return signals