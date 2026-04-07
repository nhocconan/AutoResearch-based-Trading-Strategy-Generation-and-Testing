#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: Weekly EMA(21)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Camarilla pivot levels from daily data
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r1 = close + range_hl * 1.1 / 12
    r2 = close + range_hl * 1.1 / 6
    r3 = close + range_hl * 1.1 / 4
    r4 = close + range_hl * 1.1 / 2
    s1 = close - range_hl * 1.1 / 12
    s2 = close - range_hl * 1.1 / 6
    s3 = close - range_hl * 1.1 / 4
    s4 = close - range_hl * 1.1 / 2
    
    # Volume confirmation: volume > 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        if np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend reversal OR price reaches S4 (strong support)
            if not uptrend or close[i] <= s4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend reversal OR price reaches R4 (strong resistance)
            if not downtrend or close[i] >= r4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if not volume_ok[i]:
                signals[i] = 0.0
                continue
            
            # Long: uptrend + price touches S3 (buy the dip)
            if uptrend and close[i] <= s3[i] and close[i] >= s4[i]:
                position = 1
                signals[i] = 0.25
            # Short: downtrend + price touches R3 (sell the rally)
            elif downtrend and close[i] >= r3[i] and close[i] <= r4[i]:
                position = -1
                signals[i] = -0.25
    
    return signals