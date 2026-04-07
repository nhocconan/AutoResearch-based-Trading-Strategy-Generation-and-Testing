#!/usr/bin/env python3
"""
4h Williams Alligator + Volume + 1d Trend Filter
Long when Alligator lines align bullish (jaw < teeth < lips) with above-average volume and 1d close > EMA50
Short when Alligator lines align bearish (jaw > teeth > lips) with above-average volume and 1d close < EMA50
Exit when Alligator alignment breaks
Williams Alligator identifies trend phases and filters choppy markets; volume confirms conviction; 1d EMA50 filters counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_williams_alligator_volume_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Williams Alligator (13,8,5 SMMA) ===
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(high, 13)  # Blue line (13-period SMMA of median price, simplified to high)
    teeth = smma(high, 8)  # Red line (8-period SMMA)
    lips = smma(high, 5)   # Green line (5-period SMMA)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (jaws > teeth or teeth > lips)
            if jaw[i] > teeth[i] or teeth[i] > lips[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (jaws < teeth or teeth < lips)
            if jaw[i] < teeth[i] or teeth[i] < lips[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Alligator alignment
            bullish_aligned = jaw[i] < teeth[i] and teeth[i] < lips[i]
            bearish_aligned = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Entry conditions
            if bullish_aligned and close[i] > ema_1d_50_aligned[i]:
                # Bullish alignment + above 1d EMA50 -> long
                position = 1
                signals[i] = 0.25
            elif bearish_aligned and close[i] < ema_1d_50_aligned[i]:
                # Bearish alignment + below 1d EMA50 -> short
                position = -1
                signals[i] = -0.25
    
    return signals