#!/usr/bin/env python3
"""
4h Bollinger Band Squeeze + KAMA Trend + Volume Spike
Designed to work in both bull and bear markets by:
- Bollinger Band squeeze identifies low volatility periods (pre-breakout)
- KAMA (Kaufman Adaptive Moving Average) provides trend direction
- Volume spike confirms breakout strength
- Low trade frequency target: 20-50 trades/year to minimize fee drag
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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + (bb_std * bb_std_dev)
    bb_lower = bb_mid - (bb_std * bb_std_dev)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    # KAMA (Kaufman Adaptive Moving Average) for trend
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    change = np.abs(close - np.roll(close, kama_period))
    change = np.where(np.arange(len(change)) < kama_period, 0, change)
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    volatility = pd.Series(volatility).rolling(window=kama_period, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bb_squeeze[i]) or np.isnan(kama[i]) or 
            np.isnan(bb_mid[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        squeeze = bb_squeeze[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: BB squeeze breakout above mid with volume and above KAMA
            if (price > bb_mid[i] and squeeze and vol_spike and price > kama_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: BB squeeze breakout below mid with volume and below KAMA
            elif (price < bb_mid[i] and squeeze and vol_spike and price < kama_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price closes below KAMA (trend change)
            if price < kama_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price closes above KAMA (trend change)
            if price > kama_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Bollinger_Squeeze_KAMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0