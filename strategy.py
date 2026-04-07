#!/usr/bin/env python3
"""
1h ADX-DMI Trend Strength Filter with Volume Confirmation
Long when DI+ > DI- and ADX > 25 with volume above average, short when DI- > DI+ and ADX > 25 with volume above average.
Exit when ADX falls below 20 or DI crossover reverses.
Uses ADX to filter for trending markets only, reducing whipsaws in ranging conditions.
Volume confirmation ensures breakouts have participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adx_dmi_trend_volume_v1"
timeframe = "1h"
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
    
    # === True Range and Directional Movement ===
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing (Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, 14)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hours[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Entry conditions: ADX > 25 (trending) + volume confirmation
        if adx[i] > 25 and vol_ratio[i] > 1.1:
            if plus_di[i] > minus_di[i]:
                # DI+ > DI-: bullish trend
                signals[i] = 0.20
            elif minus_di[i] > plus_di[i]:
                # DI- > DI+: bearish trend
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        # Exit: ADX < 20 (no trend) or DI crossover
        elif adx[i] < 20:
            signals[i] = 0.0
        else:
            # Hold current signal if still in trend
            if plus_di[i] > minus_di[i] and adx[i] >= 20:
                signals[i] = 0.20
            elif minus_di[i] > plus_di[i] and adx[i] >= 20:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals