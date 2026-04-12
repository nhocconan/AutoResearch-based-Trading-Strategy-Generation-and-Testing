#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_williams_vix_fix_v1
# Williams VIX Fix (WVF) measures volatility spikes from 1d data to identify exhaustion points.
# Low WVF indicates complacency (contrarian buy signal in downtrends).
# High WVF indicates panic (contrarian sell signal in uptrends).
# Combined with 6h price position relative to 20-period EMA for trend context.
# Designed for low trade frequency (target: 15-30 trades/year) to avoid fee drag.
# Works in bull markets by selling panic spikes and in bear markets by buying complacency dips.
# Focus on BTC/ETH as primary targets.

name = "6h_1d_williams_vix_fix_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Williams VIX Fix calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    # Calculate Williams VIX Fix (22-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest high in lookback period
    highest_high = pd.Series(high_1d).rolling(window=22, min_periods=22).max().values
    # WVF formula: ((highest_high - low) / highest_high) * 100
    wvf = ((highest_high - low_1d) / highest_high) * 100
    
    # Align daily WVF to 6h timeframe
    wvf_aligned = align_htf_to_ltf(prices, df_1d, wvf)
    
    # 6h EMA(20) for trend context
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(wvf_aligned[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0
            continue
        
        # Contrarian logic based on WVF extremes
        # Low WVF (< 20) = complacency -> potential buy in downtrend
        # High WVF (> 50) = panic -> potential sell in uptrend
        
        if wvf_aligned[i] < 20 and close[i] < ema_20[i] and position != 1:
            # Buy signal: complacency during downtrend
            position = 1
            signals[i] = 0.25
        elif wvf_aligned[i] > 50 and close[i] > ema_20[i] and position != -1:
            # Sell signal: panic during uptrend
            position = -1
            signals[i] = -0.25
        # Exit when WVF returns to neutral range (30-40) or opposite extreme
        elif position == 1 and (wvf_aligned[i] > 40 or close[i] >= ema_20[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (wvf_aligned[i] < 30 or close[i] <= ema_20[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals