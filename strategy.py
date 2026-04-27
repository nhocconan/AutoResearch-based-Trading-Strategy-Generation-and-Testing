#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # 4h trend: EMA21 > EMA50
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    trend_4h = ema21_aligned > ema50_aligned  # True = bullish trend
    
    # 1d trend: EMA50 > EMA200
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    trend_1d = ema50_1d_aligned > ema200_1d_aligned  # True = bullish trend
    
    # Combined trend: both 4h and 1d must agree
    combined_trend = trend_4h & trend_1d
    
    # 1h entry: Donchian breakout with volume confirmation
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Volume spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # 20% position size
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Warmup period
    start_idx = max(lookback, vol_period, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(ema21_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume + bullish trend
            if (price > highest_high[i] and 
                vol_ratio > 2.0 and 
                combined_trend[i]):
                signals[i] = size
                position = 1
            # Short entry: price breaks below Donchian low + volume + bearish trend
            elif (price < lowest_low[i] and 
                  vol_ratio > 2.0 and 
                  not combined_trend[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns bearish
            if price < lowest_low[i] or not combined_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns bullish
            if price > highest_high[i] or combined_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian20_Trend4h1d_Volume"
timeframe = "1h"
leverage = 1.0