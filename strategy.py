#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Breakout_ADX_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR and DM (Wilder's smoothing)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h Keltner Channel (20, 2.0)
    ma = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    atr = pd.Series(high - low).ewm(span=20, adjust=False, min_periods=20).mean()
    upper = ma + 2.0 * atr
    lower = ma - 2.0 * atr
    
    # Volume confirmation: current > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_1d_aligned[i]
        upper_val = upper[i]
        lower_val = lower[i]
        vol_ok = volume_ok[i]
        
        if position == 0:
            # Enter long: price above upper Keltner + strong trend (ADX>25) + volume
            if (close[i] > upper_val and 
                adx_val > 25 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price below lower Keltner + strong trend (ADX>25) + volume
            elif (close[i] < lower_val and 
                  adx_val > 25 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below middle line OR ADX weakens (<20)
            if (close[i] < ma[i] or adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above middle line OR ADX weakens (<20)
            if (close[i] > ma[i] or adx_val < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals