#!/usr/bin/env python3
# 1h_volume_weighted_sma_crossover_4h_trend
# Hypothesis: Use volume-weighted SMA (VWMA) crossovers on 1h for entries, filtered by 4h EMA trend.
# VWMA gives more weight to price action with high volume, making crossovers more significant.
# Only take longs when price > 4h EMA50 (uptrend), shorts when price < 4h EMA50 (downtrend).
# Uses 10-period VWMA for responsiveness but with volume confirmation to reduce whipsaws.
# Target: 15-37 trades/year (~60-150 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_weighted_sma_crossover_4h_trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate VWMA (Volume Weighted Moving Average) - 10 period
    # VWMA = sum(close * volume) / sum(volume)
    vol_close = close * volume
    vwma_numerator = pd.Series(vol_close).rolling(window=10, min_periods=10).sum().values
    vwma_denominator = pd.Series(volume).rolling(window=10, min_periods=10).sum().values
    vwma = np.where(vwma_denominator != 0, vwma_numerator / vwma_denominator, 0)
    
    # Regular SMA for crossover confirmation - 10 period
    sma = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(vwma[i]) or np.isnan(sma[i]) or np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: VWMA crosses below SMA OR trend turns against us
            if (vwma[i] < sma[i]) or (close[i] < ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: VWMA crosses above SMA OR trend turns against us
            if (vwma[i] > sma[i]) or (close[i] > ema_50_4h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: VWMA crosses above SMA with uptrend
            if (vwma[i] > sma[i]) and (vwma[i-1] <= sma[i-1]) and (close[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: VWMA crosses below SMA with downtrend
            elif (vwma[i] < sma[i]) and (vwma[i-1] >= sma[i-1]) and (close[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals