#!/usr/bin/env python3
"""
12h_1w_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: Trade breakouts from weekly Camarilla pivot levels with volume confirmation and 1d ADX trend filter. 
Designed for 12-30 trades/year on 12h timeframe, works in bull markets (breakouts continue) and bear markets (breakouts fail, reverse to mean).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_Breakout_Volume_Regime_v1"
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
    
    # === WEEKLY DATA FOR CAMARILLA PIVOTS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    camarilla_levels = []
    for i in range(len(high_1w)):
        if i == 0:
            camarilla_levels.append({
                'H4': np.nan, 'H3': np.nan, 'H2': np.nan, 'H1': np.nan,
                'L1': np.nan, 'L2': np.nan, 'L3': np.nan, 'L4': np.nan
            })
        else:
            ph = high_1w[i-1]
            pl = low_1w[i-1]
            pc = close_1w[i-1]
            range_val = ph - pl
            
            if range_val == 0:
                camarilla_levels.append({
                    'H4': pc, 'H3': pc, 'H2': pc, 'H1': pc,
                    'L1': pc, 'L2': pc, 'L3': pc, 'L4': pc
                })
            else:
                camarilla_levels.append({
                    'H4': pc + range_val * 1.1 / 2,
                    'H3': pc + range_val * 1.1 / 4,
                    'H2': pc + range_val * 1.1 / 6,
                    'H1': pc + range_val * 1.1 / 12,
                    'L1': pc - range_val * 1.1 / 12,
                    'L2': pc - range_val * 1.1 / 6,
                    'L3': pc - range_val * 1.1 / 4,
                    'L4': pc - range_val * 1.1 / 2
                })
    
    # Extract arrays
    H4 = np.array([x['H4'] for x in camarilla_levels])
    H3 = np.array([x['H3'] for x in camarilla_levels])
    H2 = np.array([x['H2'] for x in camarilla_levels])
    H1 = np.array([x['H1'] for x in camarilla_levels])
    L1 = np.array([x['L1'] for x in camarilla_levels])
    L2 = np.array([x['L2'] for x in camarilla_levels])
    L3 = np.array([x['L3'] for x in camarilla_levels])
    L4 = np.array([x['L4'] for x in camarilla_levels])
    
    # Align Camarilla levels to 12h timeframe
    H4_12h = align_htf_to_ltf(prices, df_1w, H4)
    H3_12h = align_htf_to_ltf(prices, df_1w, H3)
    H2_12h = align_htf_to_ltf(prices, df_1w, H2)
    H1_12h = align_htf_to_ltf(prices, df_1w, H1)
    L1_12h = align_htf_to_ltf(prices, df_1w, L1)
    L2_12h = align_htf_to_ltf(prices, df_1w, L2)
    L3_12h = align_htf_to_ltf(prices, df_1w, L3)
    L4_12h = align_htf_to_ltf(prices, df_1w, L4)
    
    # === DAILY DATA FOR ADX TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
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
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12H INDICATORS ===
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_12h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_12h[i] > 20
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Long: price breaks above H4 with volume and trend
        long_signal = (close[i] > H4_12h[i] and 
                      strong_volume and 
                      trending)
        
        # Short: price breaks below L4 with volume and trend
        short_signal = (close[i] < L4_12h[i] and 
                       strong_volume and 
                       trending)
        
        # Exit: price returns to H2/L2 or trend weakens
        exit_long = (position == 1 and 
                    (close[i] < H2_12h[i] or adx_12h[i] < 15))
        exit_short = (position == -1 and 
                     (close[i] > L2_12h[i] or adx_12h[i] < 15))
        
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