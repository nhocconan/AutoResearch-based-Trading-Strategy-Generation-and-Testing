#!/usr/bin/env python3
"""
12h_KAMA_Direction_Volume_Trend_Filter
Hypothesis: KAMA adapts to market noise, reducing whipsaw in ranging markets and capturing trends efficiently.
Combined with volume confirmation and weekly EMA trend filter, it captures institutional moves in both bull and bear markets.
Target: 12-37 trades/year (50-150 total over 4 years) to balance opportunity and fee drag.
"""

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
    volume = prices['volume'].values
    
    # 1-day data for weekly EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 1 else np.abs(np.diff(close)).sum()
    # Correct calculation for volatility (sum of absolute changes over er_length period)
    volatility = np.array([np.sum(np.abs(np.diff(close[i-er_length+1:i+1]))) if i >= er_length-1 else 0 for i in range(len(close))])
    er = np.where(volatility != 0, change / volatility, 0)
    # Pad the beginning
    er = np.concatenate([np.zeros(er_length-1), er[:len(close)-er_length+1]])
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length-1] = close[er_length-1]
    for i in range(er_length, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Weekly EMA trend filter (using daily data)
    ema_1d = pd.Series(df_1d['close']).ewm(span=35, adjust=False, min_periods=35).mean().values
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = max(er_length, 20)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(volume_filter[i]) or np.isnan(ema_1d_12h[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_12h[i]
        
        if position == 0:
            # Long: price above KAMA with volume in uptrend
            if price > kama_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price below KAMA with volume in downtrend
            elif price < kama_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day for 12h timeframe)
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price crosses below KAMA or trend reverses
                if price < kama_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day for 12h timeframe)
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price crosses above KAMA or trend reverses
                if price > kama_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_KAMA_Direction_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0