#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter + 1w EMA50 trend + Volume spike
# Uses Choppiness Index on 1d to detect trending (CHOP < 38.2) vs ranging (CHOP > 61.8) markets.
# In trending markets, follow 1w EMA50 direction with 1d Donchian(20) breakout entries.
# Volume confirmation: current volume > 2.0x 20-period average.
# Weekly EMA50 provides long-term trend filter to avoid counter-trend trades.
# Designed for low frequency (target 10-25 trades/year) to minimize fee decay in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Choppiness Index on 1d (period=14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = np.zeros(len(close_1d))
    tr = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14)
    atr_1d[0] = tr[0]
    for i in range(1, len(atr_1d)):
        if i < 14:
            atr_1d[i] = (atr_1d[i-1] * i + tr[i]) / (i + 1)
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR(14)
    sum_atr = np.zeros(len(close_1d))
    for i in range(len(sum_atr)):
        if i < 14:
            sum_atr[i] = np.sum(atr_1d[:i+1])
        else:
            sum_atr[i] = np.sum(atr_1d[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros(len(close_1d))
    lowest_low = np.zeros(len(close_1d))
    for i in range(len(highest_high)):
        if i < 14:
            highest_high[i] = np.max(high_1d[:i+1])
            lowest_low[i] = np.min(low_1d[:i+1])
        else:
            highest_high[i] = np.max(high_1d[i-13:i+1])
            lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum_atr / (highest_high - lowest_low)) / log10(14)
    chop = np.full(len(close_1d), 50.0)  # default neutral
    for i in range(14, len(close_1d)):
        if highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) on 1d
    donchian_period = 20
    upper = np.full(len(close_1d), np.nan)
    lower = np.full(len(close_1d), np.nan)
    
    for i in range(donchian_period, len(close_1d)):
        upper[i] = np.max(high_1d[i-donchian_period:i])
        lower[i] = np.min(low_1d[i-donchian_period:i])
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # 20-period average volume on 1d for spike detection
    vol_ma = np.full(len(close_1d), np.nan)
    vol_period = 20
    for i in range(vol_period, len(close_1d)):
        vol_ma[i] = np.mean(volume[i-vol_period:i])  # volume is already 1d aligned via prices index
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(donchian_period, vol_period, 14) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        # Trend direction from 1w EMA50
        uptrend = price > ema_50_1w_aligned[i]
        downtrend = price < ema_50_1w_aligned[i]
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: price breaks above upper Donchian in uptrend + volatile
            if trending_market and uptrend and price > upper_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower Donchian in downtrend + volatile
            elif trending_market and downtrend and price < lower_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend reverses
            if price < lower_aligned[i] or price < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend reverses
            if price > upper_aligned[i] or price > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Chop_Trend_EMA50_Donchian_Volume"
timeframe = "1d"
leverage = 1.0