#!/usr/bin/env python3
name = "6h_Williams_Alligator_ADX_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1H DATA FOR WILLIAMS ALLIGATOR ===
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Williams Alligator: three SMAs with future shifts
    # Jaw (blue): 13-period SMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMA, shifted 5 bars ahead  
    # Lips (green): 5-period SMA, shifted 3 bars ahead
    sma13_1h = pd.Series(close_1h).rolling(window=13, min_periods=13).mean().values
    sma8_1h = pd.Series(close_1h).rolling(window=8, min_periods=8).mean().values
    sma5_1h = pd.Series(close_1h).rolling(window=5, min_periods=5).mean().values
    
    # Apply shifts (Alligator uses future data for alignment, but we lag it properly)
    jaw_1h = np.roll(sma13_1h, 8)
    teeth_1h = np.roll(sma8_1h, 5)
    lips_1h = np.roll(sma5_1h, 3)
    
    # Invalidate the shifted values (since we can't use future data)
    jaw_1h[:8] = np.nan
    teeth_1h[:5] = np.nan
    lips_1h[:3] = np.nan
    
    # Align to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1h, jaw_1h)
    teeth_6h = align_htf_to_ltf(prices, df_1h, teeth_1h)
    lips_6h = align_htf_to_ltf(prices, df_1h, lips_1h)
    
    # === 1D ADX FOR TREND STRENGTH FILTER ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])  # Initial value
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_sum = wilder_smooth(tr, period)
    plus_dm_sum = wilder_smooth(plus_dm, period)
    minus_dm_sum = wilder_smooth(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_sum != 0, 100 * plus_dm_sum / tr_sum, 0)
    minus_di = np.where(tr_sum != 0, 100 * minus_dm_sum / tr_sum, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, period)
    
    # Align ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(adx_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 (strong trend)
            if (lips_6h[i] > teeth_6h[i] > jaw_6h[i] and adx_6h[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 (strong trend)
            elif (lips_6h[i] < teeth_6h[i] < jaw_6h[i] and adx_6h[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator lines cross (lips < teeth) OR ADX weakens
            if lips_6h[i] < teeth_6h[i] or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator lines cross (lips > teeth) OR ADX weakens
            if lips_6h[i] > teeth_6h[i] or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals