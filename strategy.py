# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume Spike + Price Rejection (Pin Bar) with 1d Trend Filter
# Enter long when: (1) Bullish pin bar (long lower wick, small body) forms at support,
# (2) Volume spike (>2x 20-period median), (3) Price above 1d EMA50 (uptrend).
# Enter short when: (1) Bearish pin bar (long upper wick, small body) forms at resistance,
# (2) Volume spike, (3) Price below 1d EMA50 (downtrend).
# Exit on opposite pin bar or volume normalization.
# Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pin bar detection
    body = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    
    # Bullish pin: long lower wick, small body, close near high
    bullish_pin = (lower_wick > 2 * body) & (body < (high - low) * 0.3) & (close > open_)
    # Bearish pin: long upper wick, small body, close near low
    bearish_pin = (upper_wick > 2 * body) & (body < (high - low) * 0.3) & (close < open_)
    
    # Volume spike (>2x 20-period median)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    volume_spike = volume > 2 * vol_median
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after warmup for volume median
        # Skip if EMA data not ready
        if np.isnan(ema_50_1d_aligned[i]):
            continue
        
        # Long entry: bullish pin + volume spike + price above 1d EMA50 (uptrend)
        if (bullish_pin[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish pin + volume spike + price below 1d EMA50 (downtrend)
        elif (bearish_pin[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite pin bar or volume normalization (volume < 1.5x median)
        elif position == 1 and (bearish_pin[i] or volume[i] < 1.5 * vol_median[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_pin[i] or volume[i] < 1.5 * vol_median[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Volume_Spin_Pin_Bar_Trend_Filter"
timeframe = "4h"
leverage = 1.0