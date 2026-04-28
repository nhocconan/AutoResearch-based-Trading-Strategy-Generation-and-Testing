#!/usr/bin/env python3
"""
4h_Keltner_Reversal_TrendFilter
Hypothesis: Keltner Channel reversals with 1d EMA200 trend filter and volume spikes capture high-probability reversals in both bull and bear markets. Uses conservative parameters to limit trades and avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: corrected import name

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_ltf_to_htf(prices, df_1d, ema_200_1d)  # Note: corrected function name
    
    # Keltner Channel: EMA20 ± 2 * ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Keltner reversal signals
        long_signal = close[i] < kc_lower[i] and close[i-1] >= kc_lower[i-1]  # Price crosses above lower band
        short_signal = close[i] > kc_upper[i] and close[i-1] <= kc_upper[i-1]  # Price crosses below upper band
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: Keltner reversal in direction of trend with volume
        long_entry = vol_confirm and uptrend and long_signal
        short_entry = vol_confirm and downtrend and short_signal
        
        # Exit logic: opposite Keltner band touch or trend change
        long_exit = close[i] > kc_upper[i] or (not uptrend)
        short_exit = close[i] < kc_lower[i] or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Keltner_Reversal_TrendFilter"
timeframe = "4h"
leverage = 1.0