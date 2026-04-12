#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_v1
Hypothesis: Trade Camarilla pivot level breakouts (H3/L3 from daily) with volume confirmation (>1.5x 20-period average) and trend filter (ADX > 25 on daily). 
Long when price breaks above H3 with volume and ADX>25, short when breaks below L3 with volume and ADX>25. Exit when price returns to PIVOT point or ADX drops below 20.
Designed for 20-30 trades/year with clear rules to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS AND ADX ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # PIVOT = (H + L + C) / 3
    # H3 = PIVOT + 1.1 * (H - L) / 2
    # L3 = PIVOT - 1.1 * (H - L) / 2
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    hl_range = high_1d - low_1d
    h3 = pivot + 1.1 * hl_range / 2
    l3 = pivot - 1.1 * hl_range / 2
    
    # Calculate ADX (14-period) for trend filter
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    plus_dm_smooth = wilders_smooth(plus_dm, period)
    minus_dm_smooth = wilders_smooth(minus_dm, period)
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, period)
    
    # Align Camarilla levels and ADX to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4H INDICATORS ===
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_aligned[i] > 25
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price breaks above H3 with volume and trend
        long_signal = (close[i] > h3_aligned[i] and 
                      strong_volume and 
                      trending)
        
        # Short: price breaks below L3 with volume and trend
        short_signal = (close[i] < l3_aligned[i] and 
                       strong_volume and 
                       trending)
        
        # Exit: price returns to PIVOT point or trend weakens
        exit_long = (position == 1 and 
                    (close[i] < pivot_aligned[i] or adx_aligned[i] < 20))
        exit_short = (position == -1 and 
                     (close[i] > pivot_aligned[i] or adx_aligned[i] < 20))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals