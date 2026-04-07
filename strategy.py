#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: On 4-hour timeframe, use Camarilla pivot levels from daily timeframe for entry/exit levels, combined with volume confirmation and ADX trend filter.
Enter long when price crosses above L4 level with volume > 1.3x 20-period average and ADX > 25.
Enter short when price crosses below H4 level with volume > 1.3x 20-period average and ADX > 25.
Exit when price crosses back below L4 (for longs) or above H4 (for shorts).
Camarilla levels provide precise support/resistance from higher timeframe; volume confirms institutional interest; ADX ensures trending conditions.
Target: 20-40 trades/year to minimize fee drag while capturing institutional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    # H2 = Close + 0.75 * (High - Low)
    # L2 = Close - 0.75 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # L1 = Close - 0.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    
    # Calculate for each day, then shift by 1 to use previous day's levels
    hl_range = d_high - d_low
    h4 = d_close + 1.5 * hl_range
    l4 = d_close - 1.5 * hl_range
    h3 = d_close + 1.125 * hl_range
    l3 = d_close - 1.125 * hl_range
    h2 = d_close + 0.75 * hl_range
    l2 = d_close - 0.75 * hl_range
    h1 = d_close + 0.5 * hl_range
    l1 = d_close - 0.5 * hl_range
    pivot = (d_high + d_low + d_close) / 3
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    h4_prev = np.roll(h4, 1)
    l4_prev = np.roll(l4, 1)
    h3_prev = np.roll(h3, 1)
    l3_prev = np.roll(l3, 1)
    h2_prev = np.roll(h2, 1)
    l2_prev = np.roll(l2, 1)
    h1_prev = np.roll(h1, 1)
    l1_prev = np.roll(l1, 1)
    pivot_prev = np.roll(pivot, 1)
    
    # Set first day's values to 0 (no previous day)
    h4_prev[0] = 0
    l4_prev[0] = 0
    h3_prev[0] = 0
    l3_prev[0] = 0
    h2_prev[0] = 0
    l2_prev[0] = 0
    h1_prev[0] = 0
    l1_prev[0] = 0
    pivot_prev[0] = 0
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_prev)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_prev)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2_prev)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2_prev)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1_prev)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1_prev)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_prev)
    
    # ADX filter on 4h to identify trending conditions
    # ADX calculation: +DM, -DM, TR, then smoothed
    period_adx = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=period_adx, min_periods=period_adx).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=period_adx, min_periods=period_adx).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=period_adx, min_periods=period_adx).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    # Avoid division by zero
    dx_denom = plus_di + minus_di
    dx = np.where(dx_denom != 0, 100 * np.abs(plus_di - minus_di) / dx_denom, 0)
    adx = pd.Series(dx).rolling(window=period_adx, min_periods=period_adx).mean().values
    
    # Volume filter: 4h volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(period_adx, n):  # Start after ADX warmup
        # Skip if any data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.3
        
        # ADX trend filter
        trending = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses back below L4
            if close[i] < l4_aligned[i]:
                exit_long = True
            # Exit when trend weakens
            elif adx[i] < 20:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses back above H4
            if close[i] > h4_aligned[i]:
                exit_short = True
            # Exit when trend weakens
            elif adx[i] < 20:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above L4 with volume and trend
            long_entry = (close[i] > l4_aligned[i] and 
                         close[i-1] <= l4_aligned[i-1] and
                         vol_confirmed and trending)
            
            # Short entry: price crosses below H4 with volume and trend
            short_entry = (close[i] < h4_aligned[i] and 
                          close[i-1] >= h4_aligned[i-1] and
                          vol_confirmed and trending)
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals