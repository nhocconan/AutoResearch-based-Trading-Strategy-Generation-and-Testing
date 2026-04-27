#!/usr/bin/env python3
"""
#100990 - 1d_WickReversal_Volume_Spike_TrendFilter
Hypothesis: Daily reversal signals based on long wicks at support/resistance with volume confirmation and weekly trend filter.
Works in both bull and bear markets by capturing exhaustion moves. Uses weekly trend to filter direction and volume spike for confirmation.
Target: 15-25 trades/year to minimize fee drag. Uses discrete position sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily body and wicks
    body = np.abs(close - open_price)
    total_range = high - low
    lower_wick = np.minimum(open_price, close) - low
    upper_wick = high - np.maximum(open_price, close)
    
    # Avoid division by zero
    lower_wick_ratio = np.where(total_range > 0, lower_wick / total_range, 0)
    upper_wick_ratio = np.where(total_range > 0, upper_wick / total_range, 0)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(lower_wick_ratio[i]) or np.isnan(upper_wick_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Bullish reversal: long lower wick (rejection of lower prices) with volume spike
        # Only take if weekly trend is up (price above weekly EMA50)
        if (lower_wick_ratio[i] > 0.6 and  # Long lower wick (>60% of range)
            volume_spike[i] and 
            close[i] > open_price[i] and   # Bullish close
            close[i] > ema50_1w_aligned[i]):  # Above weekly uptrend
            signals[i] = 0.25
            position = 1
        # Bearish reversal: long upper wick (rejection of higher prices) with volume spike
        # Only take if weekly trend is down (price below weekly EMA50)
        elif (upper_wick_ratio[i] > 0.6 and  # Long upper wick (>60% of range)
              volume_spike[i] and 
              close[i] < open_price[i] and   # Bearish close
              close[i] < ema50_1w_aligned[i]):  # Below weekly downtrend
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite wick signal or trend change
        elif position == 1 and (upper_wick_ratio[i] > 0.5 or close[i] < ema50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (lower_wick_ratio[i] > 0.5 or close[i] > ema50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WickReversal_Volume_Spike_TrendFilter"
timeframe = "1d"
leverage = 1.0