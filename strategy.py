#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
    # Donchian(20) provides clear breakout levels that work in all regimes
    # 1d ATR filter ensures we only trade when volatility is sufficient
    # Volume confirmation avoids false breakouts
    # Target: 20-50 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
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
    
    # Calculate ATR using Wilder's smoothing
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        elif i == 14:
            atr_1d[i] = np.mean(tr_1d[0:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # ATR ratio: current ATR / 20-day ATR average (volatility regime filter)
    atr_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        atr_ma_20[i] = np.mean(atr_1d[i-19:i+1])
    
    atr_ratio = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(atr_1d[i]) and not np.isnan(atr_ma_20[i]) and atr_ma_20[i] > 0:
            atr_ratio[i] = atr_1d[i] / atr_ma_20[i]
        else:
            atr_ratio[i] = np.nan
    
    # Align 1d ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    
    for i in range(n):
        if i < 19:
            donchian_h[i] = np.nan
            donchian_l[i] = np.nan
        else:
            donchian_h[i] = np.max(high[i-19:i+1])
            donchian_l[i] = np.min(low[i-19:i+1])
    
    # 4h volume spike filter (current volume > 1.5 * 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    volume_spike = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma_20[i]) and vol_ma_20[i] > 0:
            volume_spike[i] = volume[i] > 1.5 * vol_ma_20[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volatility is sufficient (ATR ratio > 0.8) and not excessive (< 2.0)
        vol_filter = (atr_ratio_aligned[i] > 0.8) and (atr_ratio_aligned[i] < 2.0)
        
        # Breakout logic with volume confirmation
        long_breakout = close[i] > donchian_h[i] and volume_spike[i] and vol_filter
        short_breakout = close[i] < donchian_l[i] and volume_spike[i] and vol_filter
        
        # Exit on opposite Donchian test or volume dropout
        long_exit = close[i] < donchian_l[i] or (not volume_spike[i])
        short_exit = close[i] > donchian_h[i] or (not volume_spike[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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

name = "4h_1d_donchian_breakout_atr_vol_filter_v1"
timeframe = "4h"
leverage = 1.0