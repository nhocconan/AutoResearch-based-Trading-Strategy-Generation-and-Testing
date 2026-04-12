#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Camarilla levels from 1d provide institutional support/resistance
    # 1w EMA50 filter ensures we only trade in direction of weekly trend
    # Volume spike confirms institutional participation
    # Target: 15-30 trades/year per symbol (60-120 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (using previous day's range)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    camarilla_pivot = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        # Pivot point
        camarilla_pivot[i] = (phigh + plow + 2 * pclose) / 4
        
        # Range
        rng = phigh - plow
        
        # Camarilla levels
        camarilla_h4[i] = camarilla_pivot[i] + rng * 1.1 / 2
        camarilla_l4[i] = camarilla_pivot[i] - rng * 1.1 / 2
        camarilla_h3[i] = camarilla_pivot[i] + rng * 1.1 / 4
        camarilla_l3[i] = camarilla_pivot[i] - rng * 1.1 / 4
        camarilla_h2[i] = camarilla_pivot[i] + rng * 1.1 / 6
        camarilla_l2[i] = camarilla_pivot[i] - rng * 1.1 / 6
        camarilla_h1[i] = camarilla_pivot[i] + rng * 1.1 / 12
        camarilla_l1[i] = camarilla_pivot[i] - rng * 1.1 / 12
    
    # Align 1d Camarilla levels to 1d timeframe (no shift needed as we use previous day's levels)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume spike filter (current volume > 2.0 * 20-day average)
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_spike = volume > 2.0 * vol_ma_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close_1d[-1] > ema_50_1w_aligned[i] if len(close_1d) > 0 else False
        weekly_downtrend = close_1d[-1] < ema_50_1w_aligned[i] if len(close_1d) > 0 else False
        
        # Long conditions: price breaks above H4 with volume spike in uptrend
        long_entry = (close[i] > h4_aligned[i] and 
                     volume_spike[i] and 
                     weekly_uptrend)
                     
        # Short conditions: price breaks below L4 with volume spike in downtrend
        short_entry = (close[i] < l4_aligned[i] and 
                      volume_spike[i] and 
                      weekly_downtrend)
        
        # Exit conditions: price returns to pivot level
        long_exit = close[i] <= pivot_aligned[i]
        short_exit = close[i] >= pivot_aligned[i]
        
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
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_vol_trend_v1"
timeframe = "1d"
leverage = 1.0