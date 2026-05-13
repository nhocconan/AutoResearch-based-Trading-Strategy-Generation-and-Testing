# 165111
#!/usr/bin/env python3
"""
6h_Elder_Ray_BullPower_Force_With_Volume_Regime
Hypothesis: Elder Ray Bull Power (High - EMA13) measures bullish force, Bear Power (Low - EMA13) measures bearish force.
Combined with volume confirmation and regime filter (ADX < 25 for range, ADX > 25 for trend), 
this captures momentum shifts in 6-hour bars. Works in bull markets via Bull Power breaks and 
in bear markets via Bear Power breaks. Low turnover expected (~15-25 trades/year).
"""

name = "6h_Elder_Ray_BullPower_Force_With_Volume_Regime"
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
    
    # Get 1d data for regime filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # EMA13 for Elder Ray calculation (6-period EMA for 6h timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation: current volume > 1.8x 20-period average (~5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # Regime filter: ADX from 1d to determine trend/range
    # Calculate ADX components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(arr[1:period]) if np.any(~np.isnan(arr[1:period])) else 0
            # Subsequent values: Wilder smoothing
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] - (result[i-1] / period) + arr[i]
                else:
                    result[i] = np.nanmean(arr[i-period+1:i+1]) if np.any(~np.isnan(arr[i-period+1:i+1])) else 0
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align regime indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup for indicators
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema13[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Bull Power positive AND increasing, volume confirmation, not in strong trend (ADX < 30)
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and  # increasing bullish force
                volume_filter[i] and 
                adx_aligned[i] < 30):  # avoid whipsaw in strong trends
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power negative AND decreasing, volume confirmation, not in strong trend
            elif (bear_power[i] < 0 and 
                  bear_power[i] < bear_power[i-1] and  # increasing bearish force (more negative)
                  volume_filter[i] and 
                  adx_aligned[i] < 30):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR volume drops OR ADX too high (trend exhaustion)
            if (bull_power[i] <= 0 or 
                not volume_filter[i] or 
                adx_aligned[i] > 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive OR volume drops OR ADX too high
            if (bear_power[i] >= 0 or 
                not volume_filter[i] or 
                adx_aligned[i] > 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals