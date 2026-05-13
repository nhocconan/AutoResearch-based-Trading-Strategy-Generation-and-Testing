#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w HMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band and close > 1w HMA50 with volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band and close < 1w HMA50 with volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 30-100 total trades over 4 years on 1d timeframe.
# 1w HMA50 ensures higher timeframe trend alignment; volume confirmation reduces false breakouts.
# Donchian channels provide clear structure; trend filter avoids counter-trend whipsaws in bear markets.

name = "1d_Donchian20_1wHMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w HMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    hma_50_1w = calculate_hma(df_1w['close'].values, 50)
    hma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_50_1w)
    
    # Calculate Donchian channels (20-period) from daily data
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(hma_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 1w HMA50, volume spike
            if (high[i] > upper[i] and 
                close[i] > hma_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 1w HMA50, volume spike
            elif (low[i] < lower[i] and 
                  close[i] < hma_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower OR volume drops below average
            if (low[i] < lower[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper OR volume drops below average
            if (high[i] > upper[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(close).ewm(span=half_period, adjust=False).mean()
    wma_full = pd.Series(close).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values