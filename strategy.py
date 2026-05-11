#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: On 12h timeframe, price breaking out of 20-period Donchian channel with 1d trend confirmation and volume expansion captures sustained momentum moves. Works in both bull and bear markets by following the higher timeframe trend. Targets 50-150 total trades over 4 years.
"""

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d EMA34 for trend ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 1d volume average for confirmation ---
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- 12h Donchian channel (20 period) ---
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Simple stoploss: 2.0x ATR from entry
                atr_est = np.abs(high_12h[i] - low_12h[i])  # rough 12h ATR estimate
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 1d average
        vol_confirm = volume_12h[i] > 1.5 * vol_avg_12h[i]
        
        if position == 0:
            # Look for breakout entries
            if vol_confirm:
                # Long breakout: price above upper Donchian + above 1d EMA34
                if close_12h[i] > highest_high[i] and close_12h[i] > ema34_12h[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_12h[i]
                # Short breakout: price below lower Donchian + below 1d EMA34
                elif close_12h[i] < lowest_low[i] and close_12h[i] < ema34_12h[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit if price breaks below lower Donchian or below 1d EMA34
                if close_12h[i] < lowest_low[i] or close_12h[i] < ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit if price breaks above upper Donchian or above 1d EMA34
                if close_12h[i] > highest_high[i] or close_12h[i] > ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals