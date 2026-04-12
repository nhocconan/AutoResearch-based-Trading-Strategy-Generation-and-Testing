#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and 1w trend filter
    # Donchian breakouts capture institutional order flow at key levels
    # 1d ATR filter ensures sufficient volatility for breakout follow-through
    # 1w trend filter aligns with higher timeframe momentum to avoid counter-trend trades
    # Works in bull/bear by only taking breakouts in direction of weekly trend
    # Target: 12-30 trades/year per symbol (60-150 over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d ATR(14)
    atr_1d = np.full(len(df_1d), np.nan)
    tr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    for i in range(13, len(df_1d)):
        if i == 13:
            atr_1d[i] = np.mean(tr_1d[0:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d ATR(30) for volatility regime filter
    atr_30_1d = np.full(len(df_1d), np.nan)
    for i in range(29, len(df_1d)):
        atr_30_1d[i] = np.mean(tr_1d[i-29:i+1])
    
    # ATR ratio: current ATR(14) / ATR(30) - identifies volatility expansion
    atr_ratio_1d = np.full(len(df_1d), np.nan)
    for i in range(29, len(df_1d)):
        if atr_30_1d[i] > 0:
            atr_ratio_1d[i] = atr_1d[i] / atr_30_1d[i]
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i == 0:
            ema_34_1w[i] = close_1w[i]
        else:
            ema_34_1w[i] = (close_1w[i] * 0.0588) + (ema_34_1w[i-1] * 0.9412)  # 2/(34+1)
    
    # Align 1d ATR ratio and 1w EMA to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    for i in range(19, n):
        donchian_h[i] = np.max(high[i-19:i+1])
        donchian_l[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above EMA34 = uptrend, below = downtrend
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volatility filter: ATR ratio > 1.2 indicates volatility expansion (breakout favorable)
        vol_expansion = atr_ratio_aligned[i] > 1.2
        
        # Donchian breakout conditions
        upper_breakout = close[i] > donchian_h[i]
        lower_breakout = close[i] < donchian_l[i]
        
        # Entry logic: breakout in direction of weekly trend with volatility expansion
        long_entry = upper_breakout and weekly_uptrend and vol_expansion
        short_entry = lower_breakout and weekly_downtrend and vol_expansion
        
        # Exit logic: opposite Donchian test or volatility contraction
        long_exit = close[i] < donchian_l[i] or (atr_ratio_aligned[i] < 0.8)
        short_exit = close[i] > donchian_h[i] or (atr_ratio_aligned[i] < 0.8)
        
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

name = "6h_1d_1w_donchian_atr_trend_v1"
timeframe = "6h"
leverage = 1.0