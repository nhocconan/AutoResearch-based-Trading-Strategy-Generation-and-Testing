#!/usr/bin/env python3
"""
6h_ADX_Alligator_Filter_1dTrend_Volume
Hypothesis: On 6h timeframe, enter long when ADX > 25 (trending), price > Alligator Jaw (EMA13) AND 1d trend is up (close > EMA34) AND volume > 1.5x 20-period average volume. Enter short when ADX > 25, price < Alligator Jaw (EMA13) AND 1d trend is down (close < EMA34) AND volume > 1.5x 20-period average volume. Exit when ADX < 20 (range) or price crosses Alligator Teeth (EMA8). Uses discrete sizing (0.0, ±0.25) to limit fee drag. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Alligator (6h timeframe): Jaw=EMA13, Teeth=EMA8, Lips=EMA5
    close_s = pd.Series(close)
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw (EMA13)
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values   # Teeth (EMA8)
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean().values    # Lips (EMA5)
    
    # Calculate ADX (6h timeframe)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, n):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = wilders_smoothing(dx, 14)
    
    # Volume confirmation: 1.5x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ADX, Alligator, volume MA, and 1d EMA warmup
    start_idx = max(14+13, 20, 34)  # ADX needs ~27, Alligator Jaw needs 13, volume MA needs 20, 1d EMA needs 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions
        adx_trending = adx[i] > 25
        adx_ranging = adx[i] < 20
        
        if position == 0:
            # Long: ADX trending, price > Jaw, 1d uptrend, volume spike
            long_signal = adx_trending and (close[i] > jaw[i]) and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            
            # Short: ADX trending, price < Jaw, 1d downtrend, volume spike
            short_signal = adx_trending and (close[i] < jaw[i]) and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ADX ranging OR price < Teeth (EMA8)
            if adx_ranging or close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ADX ranging OR price > Teeth (EMA8)
            if adx_ranging or close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_Filter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0