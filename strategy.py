#!/usr/bin/env python3
name = "6h_AdaptiveSuperTrend_1dTrend_Volume"
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
    
    # Load 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA200 for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1d ATR for volatility normalization
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6h ATR for Supertrend calculation
    tr6h_1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr6h_2 = np.abs(low[1:] - close[:-1])
    tr6h = np.concatenate([[np.inf], np.maximum(tr6h_1, tr6h_2)])
    atr_6h = pd.Series(tr6h).rolling(window=10, min_periods=10).mean().values
    
    # Adaptive Supertrend parameters based on 1d volatility
    # In high volatility: wider bands, longer trend
    # In low volatility: tighter bands, quicker reversals
    volatility_ratio = atr_1d_aligned / (pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values + 1e-10)
    # Normalize volatility ratio to [0.5, 2.0] range
    vol_factor = np.clip(volatility_ratio, 0.5, 2.0)
    
    # Base multiplier adjusted by volatility
    base_multiplier = 3.0
    adaptive_multiplier = base_multiplier * vol_factor
    
    # Calculate Supertrend upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + (adaptive_multiplier * atr_6h)
    lower_band = hl2 - (adaptive_multiplier * atr_6h)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            supertrend[i] = hl2[i]
            direction[i] = direction[i-1]
        else:
            if close[i] <= upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            else:
                upper_band[i] = hl2[i] + (adaptive_multiplier[i] * atr_6h[i])
            
            if close[i] >= lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            else:
                lower_band[i] = hl2[i] - (adaptive_multiplier[i] * atr_6h[i])
            
            if direction[i-1] == 1:
                if close[i] <= lower_band[i]:
                    direction[i] = -1
                else:
                    direction[i] = 1
            else:
                if close[i] >= upper_band[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1
            
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
    
    # Volume filter: current volume > 1.3x 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.3 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 200)  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(supertrend[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Supertrend (uptrend) AND above 1d EMA200 AND volume filter
            if close[i] > supertrend[i] and close[i] > ema_200_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend (downtrend) AND below 1d EMA200 AND volume filter
            elif close[i] < supertrend[i] and close[i] < ema_200_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below Supertrend or below 1d EMA200
            if close[i] < supertrend[i] or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above Supertrend or above 1d EMA200
            if close[i] > supertrend[i] or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals