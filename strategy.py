#!/usr/bin/env python3
# 4h_Stochastic_Pullback_TrendFilter_v1
# Hypothesis: In trending markets, price pulls back to the 40-60 range of the 14-period Stochastic oscillator before continuing the trend. This strategy uses 1-day EMA50 as the trend filter and enters on Stochastic pullbacks in the direction of the trend, with volume confirmation to filter false signals. Works in bull markets by buying pullbacks in uptrends and in bear markets by selling pullbacks in downtrends. Designed for low trade frequency (~20-40/year) to minimize fee drag.

name = "4h_Stochastic_Pullback_TrendFilter_v1"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 14-period Stochastic oscillator on 4h data
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_k = np.where((highest_high - lowest_low) != 0, 
                       (close - lowest_low) / (highest_high - lowest_low) * 100, 
                       50.0)  # Neutral when range is zero
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA50 (50) and Stochastic (14) and volume MA (20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(stoch_k[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (1.5x MA to balance signal quality and frequency)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + Stochastic pullback to 40-60 + volume
            if uptrend and 40 <= stoch_k[i] <= 60 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + Stochastic pullback to 40-60 + volume
            elif downtrend and 40 <= stoch_k[i] <= 60 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or Stochastic overbought (>80)
            if not uptrend or stoch_k[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or Stochastic oversold (<20)
            if not downtrend or stoch_k[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals