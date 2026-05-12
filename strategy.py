#!/usr/bin/env python3
# 6H_VOLATILITY_RATIO_MEAN_REVERSION
# Hypothesis: In 6h timeframe, when short-term volatility (ATR7) expands beyond long-term volatility (ATR30),
# price tends to revert to the mean (SMA50) after extreme moves. Works in both bull and bear markets
# by fading volatility spikes regardless of direction, with trend filter to avoid counter-trend trades.
# Uses mean reversion logic during high volatility regimes, which occurs in all market conditions.

name = "6H_VOLATILITY_RATIO_MEAN_REVERSION"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate True Range and ATR on primary timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Avoid division by zero
    volatility_ratio = np.where(atr30 > 0, atr7 / atr30, 1.0)
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    sma50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure ATR30 is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(atr7[i]) or np.isnan(atr30[i]) or 
            np.isnan(volatility_ratio[i]) or np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Volatility spike + price below SMA50 (mean reversion long)
            if (volatility_ratio[i] > 1.8 and 
                close[i] < sma50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Volatility spike + price above SMA50 (mean reversion short)
            elif (volatility_ratio[i] > 1.8 and 
                  close[i] > sma50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Volatility normalizes or price reaches SMA50
            if (volatility_ratio[i] < 1.2 or 
                close[i] >= sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Volatility normalizes or price reaches SMA50
            if (volatility_ratio[i] < 1.2 or 
                close[i] <= sma50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals