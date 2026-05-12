#!/usr/bin/env python3
name = "6h_PPO_Trend_WeeklyVolumeRatio_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1d and 1w data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # PPO: (12-period EMA - 26-period EMA) / 26-period EMA * 100
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    ppo = (ema_12 - ema_26) / ema_26 * 100
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w EMA200 for long-term trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume ratio: current volume / 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_avg
    
    # Signal thresholds
    ppo_long_threshold = 0.5
    ppo_short_threshold = -0.5
    vol_ratio_threshold = 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ppo[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: PPO crosses above threshold + above 1d EMA50 + above 1w EMA200 + volume spike
            if (ppo[i] > ppo_long_threshold and 
                ppo[i-1] <= ppo_long_threshold and
                close[i] > ema_50_1d_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and
                vol_ratio[i] > vol_ratio_threshold):
                signals[i] = 0.25
                position = 1
            # Short: PPO crosses below threshold + below 1d EMA50 + below 1w EMA200 + volume spike
            elif (ppo[i] < ppo_short_threshold and 
                  ppo[i-1] >= ppo_short_threshold and
                  close[i] < ema_50_1d_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and
                  vol_ratio[i] > vol_ratio_threshold):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: PPO crosses back below zero or below 1d EMA50
            if ppo[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: PPO crosses back above zero or above 1d EMA50
            if ppo[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals