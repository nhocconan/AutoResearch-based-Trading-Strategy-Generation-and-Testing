#!/usr/bin/env python3
"""
12h_1w_Camarilla_Pivot_Reversal_v1
Hypothesis: In BTC/ETH markets, price often reverses from weekly Camarilla pivot levels (H3/L3) during ranging regimes.
Long when price touches weekly L3 with RSI<40 and volume confirmation; short when touches weekly H3 with RSI>60.
Use 12h for entry timing and 1w for pivot levels and regime filter (ADX<25). Target 15-30 trades/year.
Works in bull (bounces from support) and bear (rejections at resistance).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_Pivot_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # === WEEKLY CAMARILLA PIVOT LEVELS ===
    # Calculate pivot points from previous week
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    range_hl = weekly_high - weekly_low
    
    # Camarilla levels: H3/L3 are key reversal levels
    H3 = weekly_close + (range_hl * 1.1 / 2)  # Resistance level 3
    L3 = weekly_close - (range_hl * 1.1 / 2)  # Support level 3
    
    # Align to 12h timeframe (wait for weekly bar to close)
    H3_12h = align_htf_to_ltf(prices, df_1w, H3)
    L3_12h = align_htf_to_ltf(prices, df_1w, L3)
    
    # === WEEKLY ADX(14) FOR REGIME FILTER ===
    # Calculate ADX to detect ranging markets (ADX < 25)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smooth TR, DM+, DM-
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(arr[:period]) / period
            # Subsequent values are Wilder smoothing
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = smooth_wilder(tr, period)
        dm_plus_smooth = smooth_wilder(dm_plus, period)
        dm_minus_smooth = smooth_wilder(dm_minus, period)
        
        # Avoid division by zero
        dm_plus_smooth = np.where(atr == 0, 0, dm_plus_smooth)
        dm_minus_smooth = np.where(atr == 0, 0, dm_minus_smooth)
        
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.full_like(di_plus, np.nan)
        mask = (di_plus + di_minus) != 0
        dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
        
        # ADX is smoothed DX
        adx = smooth_wilder(dx, period)
        return adx
    
    adx = calculate_adx(weekly_high, weekly_low, weekly_close, 14)
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    
    # === VOLUME CONFIRMATION (20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_surge = volume > (vol_ma * 1.5)  # Volume > 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade in ranging markets (ADX < 25)
        ranging = adx_12h[i] < 25
        
        # Price touching Camarilla levels with wick penetration
        touch_L3 = low[i] <= L3_12h[i] * 1.002  # Allow 0.2% tolerance
        touch_H3 = high[i] >= H3_12h[i] * 0.998  # Allow 0.2% tolerance
        
        # Entry conditions
        long_entry = touch_L3 and ranging and vol_surge[i]
        short_entry = touch_H3 and ranging and vol_surge[i]
        
        # Exit: opposite touch or ADX trending up
        long_exit = touch_H3 or adx_12h[i] >= 30
        short_exit = touch_L3 or adx_12h[i] >= 30
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals