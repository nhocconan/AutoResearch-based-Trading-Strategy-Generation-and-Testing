#!/usr/bin/env python3
"""
1d_Fisher_Transform_MeanReversion_v1
Hypothesis: The Fisher Transform identifies turning points in price by converting prices into a Gaussian normal distribution.
In mean-reverting markets (range-bound), extreme Fisher values (>1.5 or <-1.5) signal reversals.
In trending markets, we filter trades using weekly EMA to avoid counter-trend trades.
Targets 15-25 trades/year on 1d timeframe with low frequency to minimize fee drag.
Works in both bull and bear markets by adapting to regime: mean reversion in range, trend-following in strong trends.
"""

name = "1d_Fisher_Transform_MeanReversion_v1"
timeframe = "1d"
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
    
    # === Weekly EMA for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily Fisher Transform (Ehlers) ===
    # Lookback period for normalization
    lookback = 10
    # Initialize arrays
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    price_norm = np.full(n, np.nan)
    fish = np.full(n, np.nan)
    
    # Calculate rolling max/min for normalization
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
        # Avoid division by zero
        if highest_high[i] != lowest_low[i]:
            price_norm[i] = ((close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])) * 2 - 1
        else:
            price_norm[i] = 0
        # Clamp to avoid domain errors in log
        price_norm[i] = max(min(price_norm[i], 0.999), -0.999)
    
    # Fisher Transform calculation with smoothing
    fish = np.full(n, np.nan)
    fish_prev = 0
    for i in range(lookback-1, n):
        if np.isnan(price_norm[i]):
            continue
        # Fish = 0.5 * ln((1+price_norm)/(1-price_norm)) + 0.5 * fish_prev
        fish_val = 0.5 * np.log((1 + price_norm[i]) / (1 - price_norm[i])) + 0.5 * fish_prev
        fish[i] = fish_val
        fish_prev = fish_val
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(fish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long when Fisher is extremely negative (oversold) and price above weekly EMA (uptrend bias)
            # Enter short when Fisher is extremely positive (overbought) and price below weekly EMA (downtrend bias)
            if fish[i] < -1.5 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif fish[i] > 1.5 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Fisher crosses above -0.5 (mean reversion complete) or flip to short signal
            if fish[i] > -0.5 or (fish[i] > 1.5 and close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Exit short when Fisher crosses below 0.5 (mean reversion complete) or flip to long signal
            if fish[i] < 0.5 or (fish[i] < -1.5 and close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals