#!/usr/bin/env python3
# 6h_MultiTimeframe_Stochastic_Signal_with_12hTrend
# Hypothesis: Use 6h Stochastic oscillator for mean-reversion entries (oversold/overbought) but only in the direction of 12h trend.
# Long when 6h Stochastic < 20 and 12h close > 12h EMA50 (uptrend).
# Short when 6h Stochastic > 80 and 12h close < 12h EMA50 (downtrend).
# Exit when Stochastic crosses back above 50 (long) or below 50 (short) to avoid overstaying.
# Uses volume confirmation to avoid low-liquidity whipsaws.
# Targets 15-25 trades/year to minimize fee drag.

name = "6h_MultiTimeframe_Stochastic_Signal_with_12hTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 6h Stochastic (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    
    # Avoid division by zero
    denom = highest_high - lowest_low
    denom = np.where(denom == 0, 1e-10, denom)
    
    k_percent = 100 * ((close - lowest_low) / denom)
    
    # Smooth K to get D (3-period SMA of K)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(d_percent[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema12h_trend = ema50_12h_aligned[i]
        stoch_d = d_percent[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Oversold Stochastic + 12h uptrend + volume confirmation
            if stoch_d < 20 and close[i] > ema12h_trend and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought Stochastic + 12h downtrend + volume confirmation
            elif stoch_d > 80 and close[i] < ema12h_trend and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic crosses back above 50 (mean reversion complete)
            if stoch_d > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Stochastic crosses back below 50 (mean reversion complete)
            if stoch_d < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals