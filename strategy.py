#!/usr/bin/env python3
"""
6h_WilliamsVixFix_Trend_Filter_1d
Hypothesis: Williams Vix Fix (WVF) identifies volatility spikes and potential bottoms in bear markets.
Combined with 1d EMA50 trend filter to trade in direction of higher timeframe trend.
Goes long when WVF > 0.8 (extreme fear) and price > 1d EMA50 (uptrend).
Goes short when WVF > 0.8 and price < 1d EMA50 (downtrend).
Uses discrete sizing (0.25) to minimize fees. Target: 12-30 trades/year.
Works in bull via trend continuation after pullbacks, in bear via mean reversion at volatility extremes.
"""

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Vix Fix on 6h data
    # WVF = ((Highest High in n-period - Low) / (Highest High in n-period - Lowest Low in n-period)) * 100
    # Normalized to 0-1 range where >0.8 indicates extreme fear
    lookback = 22  # ~1 month of 6h bars
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    wvf = np.where(hh_ll != 0, ((highest_high - low) / hh_ll) * 100, 0)
    # Normalize to 0-1
    wvf_normalized = wvf / 100.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(wvf_normalized[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: extreme fear (WVF > 0.8) and uptrend (price > daily EMA50)
            long_signal = (wvf_normalized[i] > 0.8) and (close[i] > ema_50_1d_aligned[i])
            # Short: extreme fear (WVF > 0.8) and downtrend (price < daily EMA50)
            short_signal = (wvf_normalized[i] > 0.8) and (close[i] < ema_50_1d_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when fear subsides (WVF < 0.5) or trend changes
            exit_signal = (wvf_normalized[i] < 0.5) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when fear subsides (WVF < 0.5) or trend changes
            exit_signal = (wvf_normalized[i] < 0.5) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_Trend_Filter_1d"
timeframe = "6h"
leverage = 1.0