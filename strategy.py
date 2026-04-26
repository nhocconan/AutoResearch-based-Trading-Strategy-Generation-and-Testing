#!/usr/bin/env python3
"""
6h_WilliamsVIXFix_Reversion_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, use Williams VIX Fix (WVF) to identify extreme fear/greed reversals, filtered by 1d trend (price > EMA34) and volume confirmation (>1.5x 20-period average). Enter long when WVF > 0.8 (extreme fear) in 1d uptrend with volume confirmation. Enter short when WVF < 0.2 (extreme greed) in 1d downtrend with volume confirmation. Uses discrete position size 0.25. Designed for 12-25 trades/year on 6h by requiring extreme readings and trend alignment, reducing whipsaw in choppy markets while capturing mean reversion in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams VIX Fix and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams VIX Fix: WVF = ((HighestClose - Low) / (HighestClose - LowestLow)) * 100
    # where HighestClose = highest close over lookback period, LowestLow = lowest low over lookback
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    wvf = ((highest_close - low) / (highest_close - lowest_low + 1e-10)) * 100
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need WVF lookback, EMA warmup, volume MA warmup
    start_idx = max(lookback, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(wvf[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: WVF > 0.8 (extreme fear) + 1d uptrend + volume confirmation
            long_signal = (wvf[i] > 80.0) and trend_1d_uptrend and volume_confirm[i]
            
            # Short: WVF < 0.2 (extreme greed) + 1d downtrend + volume confirmation
            short_signal = (wvf[i] < 20.0) and trend_1d_downtrend and volume_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: WVF < 0.5 (fear subsided) OR 1d trend turns down
            if (wvf[i] < 50.0 or not trend_1d_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: WVF > 0.5 (greed subsided) OR 1d trend turns up
            if (wvf[i] > 50.0 or not trend_1d_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVIXFix_Reversion_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0