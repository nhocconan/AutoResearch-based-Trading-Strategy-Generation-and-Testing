#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Reduced minimum to ensure sufficient data for warmup
        return np.zeros(n)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend context and signal generation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian(20) channels
    high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate 1d volume moving average (20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Get 1w data for regime filter (choppiness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ATR for choppiness
    tr1w = np.abs(high_1w - low_1w)
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1w[0] = tr2w[0] = tr3w[0] = np.nan
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_1w[i] = np.mean(trw[i-14:i+1])
    
    # Calculate 14-period high-low range for choppiness
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
    sum_atr_14 = np.full(len(df_1w), np.nan)
    range_hl_14 = np.full(len(df_1w), np.nan)
    chop = np.full(len(df_1w), 50.0)  # Default to neutral
    
    for i in range(14, len(df_1w)):
        sum_atr_14[i] = np.sum(trw[i-14:i+1])
        range_hl_14[i] = max_high_14[i] - min_low_14[i]
        if range_hl_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / range_hl_14[i]) / np.log10(14)
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start from 200 to ensure all indicators are ready
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(high_20_1d_aligned[i]) or np.isnan(low_20_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR > 0.3 * its 20-period MA (avoid low volatility)
        atr_ma_20_1d = np.full(len(df_1d), np.nan)
        for j in range(34, len(df_1d)):  # 14 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_1d[j-19:j+1])):
                atr_ma_20_1d[j] = np.mean(atr_1d[j-19:j+1])
        atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
        vol_filter = (not np.isnan(atr_ma_20_1d_aligned[i]) and 
                     atr_1d_aligned[i] > 0.3 * atr_ma_20_1d_aligned[i])
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = volume[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Regime filter: choppiness < 50 (trending market)
        trending_regime = chop_aligned[i] < 50
        
        # Trend filter: price above/below 1d Donchian mid
        mid_1d = (high_20_1d_aligned[i] + low_20_1d_aligned[i]) / 2
        uptrend = close[i] > mid_1d
        downtrend = close[i] < mid_1d
        
        # Entry conditions: 12h breakout of 1d Donchian in trend direction + volatility + volume + regime
        long_entry = (close[i] > high_20_1d_aligned[i]) and uptrend and vol_filter and vol_spike and trending_regime
        short_entry = (close[i] < low_20_1d_aligned[i]) and downtrend and vol_filter and vol_spike and trending_regime
        
        # Exit conditions: opposite 12h breakout of 1d Donchian or volatility drop or regime change
        long_exit = (close[i] < low_20_1d_aligned[i]) or (not vol_filter) or (not trending_regime)
        short_exit = (close[i] > high_20_1d_aligned[i]) or (not vol_filter) or (not trending_regime)
        
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

name = "12h_1d_1w_donchian_vol_vol_chop"
timeframe = "12h"
leverage = 1.0