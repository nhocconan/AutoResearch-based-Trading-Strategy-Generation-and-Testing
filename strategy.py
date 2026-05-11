#!/usr/bin/env python3
"""
12h_Donchian_20_Volume_Trend
Hypothesis: On 12h timeframe, enter long when price breaks above 20-period Donchian high with volume confirmation and 1d trend alignment. Enter short on breakdown below 20-period Donchian low with volume confirmation and 1d trend alignment. Uses volume spike (>1.5x 20-period average) and 1d EMA50 trend filter to avoid false breakouts. Designed for 15-30 trades/year per symbol to minimize fee drag while capturing significant trends in both bull and bear markets.
"""

name = "12h_Donchian_20_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 12h Donchian Channels (20 period) ---
    # Highest high of last 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # --- 12h Volume Spike Filter ---
    # Average volume of last 20 periods
    avg_volume = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for Donchian and EMA calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Exit on opposite Donchian breach
                if position == 1 and close_12h[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        if position == 0:
            # Look for breakout/breakdown with volume confirmation and trend alignment
            # Long: price breaks above Donchian high with volume spike and above 1d EMA50
            if (close_12h[i] > donchian_high[i] and 
                volume_spike[i] and 
                close_12h[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and below 1d EMA50
            elif (close_12h[i] < donchian_low[i] and 
                  volume_spike[i] and 
                  close_12h[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Manage existing position: exit on opposite Donchian breach
            if position == 1:
                if close_12h[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close_12h[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals