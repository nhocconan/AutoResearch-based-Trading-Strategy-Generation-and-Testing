#!/usr/bin/env python3
"""
12h_wick_volume_reversal_v1
Hypothesis: 12h price action shows rejection at key levels via long/short wicks with volume confirmation.
In bull markets: long wick rejection at support + volume = long setup.
In bear markets: short wick rejection at resistance + volume = short setup.
Uses 1d trend filter to avoid counter-trend trades. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_wick_volume_reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 12h body and wick calculations
    body = np.abs(close - open_price)
    upper_wick = high - np.maximum(close, open_price)
    lower_wick = np.minimum(close, open_price) - low
    
    # Wick strength: wick as % of total range (high-low)
    total_range = high - low
    upper_wick_pct = np.where(total_range > 0, upper_wick / total_range, 0)
    lower_wick_pct = np.where(total_range > 0, lower_wick / total_range, 0)
    
    # Volume confirmation: 24-period average (2 days of 12h data)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    # 1d trend filter: 50 EMA on close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_1d)):
        if np.isnan(ema_50[i-1]):
            ema_50[i] = close_1d[i]
        else:
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not ready
        if (np.isnan(upper_wick_pct[i]) or np.isnan(lower_wick_pct[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below recent low or trend turns down
            if low[i] < np.min(low[i-12:i]) or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above recent high or trend turns up
            if high[i] > np.max(high[i-12:i]) or not trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: long wick rejection at support with volume and uptrend
            if (lower_wick_pct[i] > 0.6 and  # Long lower wick >60% of range
                vol_ratio > 1.8 and          # Strong volume
                trend_up):                   # Uptrend filter
                position = 1
                signals[i] = 0.25
            # Short: short wick rejection at resistance with volume and downtrend
            elif (upper_wick_pct[i] > 0.6 and  # Long upper wick >60% of range
                  vol_ratio > 1.8 and          # Strong volume
                  trend_down):                 # Downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals