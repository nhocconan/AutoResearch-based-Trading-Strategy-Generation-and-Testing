#!/usr/bin/env python3
"""
12H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_1W_TREND_FILTER
Hypothesis: Use 20-period Donchian breakout on 12h timeframe with weekly trend filter and volume confirmation.
In weekly uptrend (price above 100-period SMA), only take longs from upper band breakout.
In weekly downtrend (price below 100-period SMA), only take shorts from lower band breakout.
Volume spike (2.0x 20-period) confirms breakout strength. Target 20-30 trades/year (80-120 total).
Designed for low-frequency, high-conviction trades to minimize fee drag while capturing strong trends.
Works in bull (catch breakouts) and bear (short breakdowns) markets.
"""
name = "12H_DONCHIAN_BREAKOUT_VOLUME_CONFIRMATION_1W_TREND_FILTER"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Weekly 100-period SMA for trend filter
    weekly_sma = pd.Series(df_1w['close'].values).rolling(window=100, min_periods=100).mean().values
    weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(20, n):  # Start after warmup for Donchian and volume MA
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(weekly_sma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # LONG: Weekly uptrend (price above weekly SMA) + break above upper Donchian + volume spike
            if (close[i] > weekly_sma_aligned[i] and 
                close[i] > high_roll[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Weekly downtrend (price below weekly SMA) + break below lower Donchian + volume spike
            elif (close[i] < weekly_sma_aligned[i] and 
                  close[i] < low_roll[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (below upper band)
            if close[i] < high_roll[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (above lower band)
            if close[i] > low_roll[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals