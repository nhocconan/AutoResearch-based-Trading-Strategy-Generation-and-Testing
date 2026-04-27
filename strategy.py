#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume spike
# Elder Ray measures bull/bear power relative to EMA13. Works in bull/bear:
# - Bull market: Buy when Bull Power > 0 and rising + price > EMA13
# - Bear market: Sell when Bear Power < 0 and falling + price < EMA13
# Volume spike filters weak moves. Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray calculation
    close_1d = pd.Series(df_1d['close'].values)
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1d EMA50 for trend filter
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bull Power = High - EMA13
    bull_power = high - ema13_1d_aligned
    # Bear Power = Low - EMA13 (negative value indicates bear strength)
    bear_power = low - ema13_1d_aligned
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Bull Power positive AND rising AND price above EMA50 (uptrend) + volume
        if (bull_power[i] > 0 and 
            bull_power[i] > bull_power[i-1] and  # Rising bull power
            close[i] > ema50_1d_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Bear Power negative AND falling AND price below EMA50 (downtrend) + volume
        elif (bear_power[i] < 0 and 
              bear_power[i] < bear_power[i-1] and  # Falling bear power (more negative)
              close[i] < ema50_1d_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0