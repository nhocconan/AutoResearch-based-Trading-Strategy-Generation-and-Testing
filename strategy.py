#!/usr/bin/env python3
# 6h_KAMA_Trend_Filter_PriceAction
# Hypothesis: KAMA adapts to market noise, providing a dynamic trend filter that reduces whipsaws in choppy markets.
# Combined with price action (higher highs/lows) and volume confirmation, it captures sustained trends while avoiding false signals.
# Designed for 6h timeframe to balance responsiveness and noise reduction, working in both bull and bear markets.

name = "6h_KAMA_Trend_Filter_PriceAction"
timeframe = "6h"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = np.diff(volatility, prepend=volatility[0])
        er = change / (volatility + 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate price action: higher highs and higher lows for uptrend, lower highs and lower lows for downtrend
    def price_action_trend(high, low, lookback=5):
        hh = np.zeros_like(high, dtype=bool)
        ll = np.zeros_like(low, dtype=bool)
        for i in range(lookback, len(high)):
            hh[i] = high[i] > np.max(high[i-lookback:i])
            ll[i] = low[i] < np.min(low[i-lookback:i])
        return hh, ll
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    kama_1d = kama(df_1d['close'].values)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Price action on 6h
    hh, ll = price_action_trend(high, low, lookback=3)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            np.isnan(hh[i]) or np.isnan(ll[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA(1d) + higher high + volume confirmation
            if close[i] > kama_1d_aligned[i] and hh[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA(1d) + lower low + volume confirmation
            elif close[i] < kama_1d_aligned[i] and ll[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below KAMA(1d) or lower low forms
            if close[i] < kama_1d_aligned[i] or ll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above KAMA(1d) or higher high forms
            if close[i] > kama_1d_aligned[i] or hh[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals