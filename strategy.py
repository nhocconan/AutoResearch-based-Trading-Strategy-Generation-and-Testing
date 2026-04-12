#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pullback_v1
Hypothesis: Trade pullbacks to Camarilla pivot levels (L3/H3) in trending markets (ADX > 25) with volume confirmation. 
Long when price pulls back to L3 in uptrend, short when pulls back to H3 in downtrend. 
Uses 1-day Camarilla levels calculated from prior day's range. Designed for 20-35 trades/year with clear mean-reversion within trend logic that works in bull (buy dips) and bear (sell rallies) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pullback_v1"
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
    
    # === DAILY DATA FOR CAMARILLA PIVOTS AND ADX TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Camarilla levels (based on prior day's range)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    
    # Calculate ADX (14-period) for trend strength
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
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4H INDICATORS ===
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price pulls back to L3 in uptrend
        long_signal = (strong_trend and 
                      strong_volume and 
                      close[i] <= camarilla_l3_aligned[i] * 1.002 and  # Allow small buffer
                      close[i] >= camarilla_l3_aligned[i] * 0.998)
        
        # Short: price pulls back to H3 in downtrend
        short_signal = (strong_trend and 
                       strong_volume and 
                       close[i] >= camarilla_h3_aligned[i] * 0.998 and  # Allow small buffer
                       close[i] <= camarilla_h3_aligned[i] * 1.002)
        
        # Exit: price moves back toward VWAP or trend weakens
        exit_long = (position == 1 and 
                    (close[i] >= camarilla_l3_aligned[i] * 1.01 or adx_aligned[i] < 20))
        exit_short = (position == -1 and 
                     (close[i] <= camarilla_h3_aligned[i] * 0.99 or adx_aligned[i] < 20))
        
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