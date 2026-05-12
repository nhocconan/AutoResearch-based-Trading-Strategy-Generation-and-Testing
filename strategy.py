#!/usr/bin/env python3
# 6h_RankMomentum_20_50_Trend
# Hypothesis: Rank-based momentum filtering using cross-sectional ranking of 20-period returns
# against 50-period median to identify persistent trends. Combines with weekly trend filter
# (price above/below weekly 200 EMA) and volume confirmation. The rank mechanism adapts to
# changing volatility regimes and avoids lookback window sensitivity. Designed for 6H timeframe
# to capture multi-day momentum while limiting trade frequency. Works in bull/bear by following
# higher timeframe trend direction.

name = "6h_RankMomentum_20_50_Trend"
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
    
    # === Weekly Trend Filter (200 EMA) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === Rank Momentum (20 vs 50) ===
    # 20-period returns
    returns_20 = np.zeros_like(close)
    returns_20[20:] = (close[20:] - close[:-20]) / close[:-20]
    # 50-period median of returns for adaptive threshold
    returns_50_median = pd.Series(returns_20).rolling(window=50, min_periods=50).median().values
    # Rank signal: 1 if return > median, -1 if return < median, 0 otherwise
    rank_signal = np.sign(returns_20 - returns_50_median)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(rank_signal[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Positive rank momentum, weekly uptrend, volume confirmation
            if rank_signal[i] > 0 and weekly_uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Negative rank momentum, weekly downtrend, volume confirmation
            elif rank_signal[i] < 0 and weekly_downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Rank turns negative or weekly trend breaks
            if rank_signal[i] < 0 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Rank turns positive or weekly trend breaks
            if rank_signal[i] > 0 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals