#!/usr/bin/env python3
"""
6h_Range_Reversal_at_Liquidity_Pools
Hypothesis: Price reverses at prior day's high/low liquidity pools when RSI shows exhaustion.
Works in both bull/bear markets by fading extremes with mean reversion logic.
Targets 15-25 trades/year on 6h timeframe to minimize fee drag.
"""

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
    
    # Get 1d data for liquidity pools (prior day high/low)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Prior day high and low as liquidity pools
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    
    # Align to 6h timeframe
    liquidity_high = align_htf_to_ltf(prices, df_1d, prev_day_high)
    liquidity_low = align_htf_to_ltf(prices, df_1d, prev_day_low)
    
    # RSI(14) for exhaustion detection
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: avoid low-volume false signals
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20 * 0.5)  # At least 50% of average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(liquidity_high[i]) or np.isnan(liquidity_low[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price at liquidity low (support) + RSI oversold
        near_support = low[i] <= liquidity_low[i] * 1.002  # Within 0.2% of liquidity low
        rsi_oversold = rsi_values[i] < 30
        
        # Short setup: price at liquidity high (resistance) + RSI overbought
        near_resistance = high[i] >= liquidity_high[i] * 0.998  # Within 0.2% of liquidity high
        rsi_overbought = rsi_values[i] > 70
        
        # Entry conditions with volume filter
        long_entry = near_support and rsi_oversold and volume_filter[i]
        short_entry = near_resistance and rsi_overbought and volume_filter[i]
        
        # Exit when price moves away from liquidity level or RSI normalizes
        long_exit = (close[i] > liquidity_low[i] * 1.01) or (rsi_values[i] > 50)
        short_exit = (close[i] < liquidity_high[i] * 0.99) or (rsi_values[i] < 50)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Close long
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.25   # Close short
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Range_Reversal_at_Liquidity_Pools"
timeframe = "6h"
leverage = 1.0